#!/usr/bin/env python3
import numpy as np
from dataclasses import dataclass
from utils.spline import Spline
import numba
import cv2
# @dataclass
# class Waypoint:
#     x: float
#     y: float
#     theta: float
#     left: float
#     right: float
# @numba.jit
# def get_closest_waypoint():
#     dist = np.linalg.norm(self.centerline_xy - np.array([x, y]), axis=1)
#     if n == 1:
#         ind = np.argmin(dist)
#     else:
#         dist_sorted = np.argsort(dist)
#         ind = dist_sorted[:np.min(n, len(dist_sorted))]
#     return self.centerline_points[ind].reshape(n, -1), ind # 

class Track:
    def __init__(self, centerline_points: str, map:str, initialized: bool = False):
        self.centerline_points = self.load_waypoints(centerline_points) # (N, 5) [x, y, left, right, theta]
        self.centerline_xy = self.centerline_points[:, :2]
        self.step = 0.05 # step size
        self.half_width = 0.6 # TODO: Original RLMPC code assumes uniform track width. Modify to allow for varying track width
        self.k = 3
        self.map = cv2.imread(map, cv2.IMREAD_GRAYSCALE)
        self.resulution = 0.05
        self.origin = np.array([0,0,0])
        if not initialized:
            self.x_spline: Spline = None
            self.y_spline: Spline = None
            self.length = 0.0
        else:
            self.x_spline = Spline(self.centerline_points[:, 4], self.centerline_points[:, 0], k=self.k)
            self.y_spline = Spline(self.centerline_points[:, 4], self.centerline_points[:, 1], k=self.k)
            self.length = self.centerline_points[-1, 4]

    def reset_starting_point_new(self, x: float, y: float, refine: bool = True, dsamp=25):
        """
        The order of the waypoints should be rearranged, 
        so that the closest waypoint to the starting position is the first waypoint.
        
        NOTE:
        (1) if calling this function we assume the centerline_point initially is not a closed loop, 
            hence need to +1 in size and add starting point to the end
        """
        ## Rearange waypoints
        
        N = self.centerline_points.shape[0] # NOTE: (1)
        new_points = np.zeros((N, 5))
        _, starting_index = self.get_closest_waypoint(x, y)
        new_points[:N, :4] = np.roll(self.centerline_points[:, :4], -starting_index, axis=0)
        # new_points[N, :4] = new_points[0, :4] # CLOSE THE LOOP
        ## Find s for each point
        new_points[:, 4] = self.get_cum_distance(new_points[:, :2])
        self.centerline_points = new_points
        self.centerline_xy = self.centerline_points[:, :2]
        self.length = self.centerline_points[-1, 4]
        ## Fit spline y = f(s), x = f(s)
        spline_s = np.append(self.centerline_points[::dsamp, 4], self.length+0.05)
        spline_x = np.append(self.centerline_points[::dsamp, 0], self.centerline_points[0, 0])
        spline_y = np.append(self.centerline_points[::dsamp, 1], self.centerline_points[0, 1])
        self.x_spline = Spline(spline_s, spline_x, k=self.k)
        self.y_spline = Spline(spline_s, spline_y, k=self.k)
        if refine:
            self.refine_uniform_waypoint()
        np.savetxt("map/refined_centerline.csv", self.centerline_points, delimiter=",")
        np.savetxt("map/race3_centerline.csv", self.centerline_points[:, :-1], delimiter=",")
        return

    def reset_starting_point(self, x: float, y: float, refine: bool = True):
        """
        The order of the waypoints should be rearranged, 
        so that the closest waypoint to the starting position is the first waypoint.
        
        NOTE:
        (1) if calling this function we assume the centerline_point initially is not a closed loop, 
            hence need to +1 in size and add starting point to the end
        """
        ## Rearange waypoints
        N = self.centerline_points.shape[0] # NOTE: (1)
        new_points = np.zeros((N, 5))
        _, starting_index = self.get_closest_waypoint(x, y)
        new_points[:N, :4] = np.roll(self.centerline_points[:, :4], -starting_index, axis=0)
        # new_points[N, :4] = new_points[0, :4] # CLOSE THE LOOP
        ## Find s for each point
        new_points[:, 4] = self.get_cum_distance(new_points[:, :2])
        self.centerline_points = new_points
        self.centerline_xy = self.centerline_points[:, :2]
        self.length = self.centerline_points[-1, 4]
        ## Fit spline y = f(s), x = f(s)
        
        self.x_spline = Spline(self.centerline_points[:, 4], self.centerline_points[:, 0], k=self.k)
        self.y_spline = Spline(self.centerline_points[:, 4], self.centerline_points[:, 1], k=self.k)
        if refine:
            self.refine_uniform_waypoint()
        np.savetxt("map/refined_race3_centerline.csv", self.centerline_points[:, :-1], delimiter=",")
        return
    
    def plot_spline(self):
        """
        Plot the centerline spline
        """
        import matplotlib.pyplot as plt
        s = np.linspace(0, self.length, 1000)
        xs = self.x_spline(s)
        ys = self.y_spline(s)
        plt.plot(xs, ys)
        plt.axis("equal")
        plt.show()
        return

    def refine_uniform_waypoint(self):
        """
        Sample new set of waypoints uniformly to step_size.
        """
        s = np.arange(0, self.length, self.step)
        x = self.x_spline(s)
        y = self.y_spline(s)
        
        # x = np.append(x, x[0]) # close the loop
        # y = np.append(y, y[0]) # close the loop
        points = np.hstack([x[:, None], y[:, None]])
        # s_new = self.get_cum_distance(points) # (N)
        # self.length = s_new[-1]
        
        left_right_distance = self.update_half_width(points) # TODO: UPDATE HALF DISTANCE ON S
        
        new_points = np.hstack([points[:], left_right_distance[:], s[:, None]])
        self.centerline_points = new_points
        self.centerline_xy = self.centerline_points[:, :2]
        # print("NEW LAP LENGTH: ", self.length, s_new[-2])

    def load_waypoints(self, csv_path: str):
        self.waypoints = csv_path
        with open(csv_path, "r") as f:
            waypoint = np.loadtxt(f, delimiter=",") # [[x, y, left, right], ...]
        return waypoint

    def get_closest_waypoint(self, x: float, y: float, n: int = 1):
        """
        Find Closest n waypoints to the given x, y position
        
        Return: (n, 5) [[x, y, theta, left, right], ...], index
        """
        dist = np.linalg.norm(self.centerline_xy - np.array([x, y]), axis=1)
        # print(dist.shape)
        if n == 1:
            ind = np.argmin(dist)
        else:
            dist_sorted = np.argsort(dist)
            # print(n ,len(dist_sorted))
            ind = dist_sorted[:min(n, len(dist_sorted))]
            # print(ind)
        return self.centerline_points[ind].reshape(n, -1).copy(), ind # 

    # def get_theta(self, x: float, y: float) -> float:
    #     """
    #     Given a position, estimate progress on track
    #     """
    #     closest_points, _ = self.find_closest_waypoint(x, y)
    #     # print(closest_points)
    #     return closest_points[0, 4]
    def get_states(self, x:float, y:float, yaw:float):
        """
        Get required states: [epsi, s, ey] based on xy
        """
        closest_points, ind = self.get_closest_waypoint(x, y, 2)
        # print("@=> Closest points: ", ind)
        s = self.get_theta(np.array([x, y]))
        e_y, e_yaw, s = self.global_to_track(x, y, yaw, s)
        ## Find yaw from centerline
        
        return e_yaw, s, e_y, closest_points[0, :2]
    
    def get_theta(self, point:np.ndarray) -> float:
        """
        Given a position, estimate progress s on track
        
        Use 2 closest waypoints to estimate theta
        
        Input:  [x, y]
        Return: s
        """
        # point = np.array(point)
        N = self.centerline_points.shape[0]
        x0, y0 = point[0], point[1]
        closest_points, ind = self.get_closest_waypoint(x0, y0, 2)
        if (ind[0] < N / 4 and ind[1] > 2 * N / 3):
            closest_points[0, 4] = self.length + closest_points[0, 4]
        elif (ind[1] < N / 4 and ind[0] > 2 * N / 3):
            closest_points[1, 4] = self.length + closest_points[1, 4]
        
        x1, y1, s1 = closest_points[0, [0, 1, 4]]
        x2, y2, s2 = closest_points[1, [0, 1, 4]]

        A = np.array([[x2 - x1, y2 - y1], [y2 - y1, -(x2 - x1)]])
        b = np.array([(x2 - x1) * x0 + (y2 - y1) * y0, (x2 - x1) * y1 - (y2 - y1) * x1])
        xc, yc = np.linalg.inv(A).dot(b)
        d_1c = np.linalg.norm(np.array([xc, yc]) - np.array([x1, y1]))
        d_2c = np.linalg.norm(np.array([xc, yc]) - np.array([x2, y2]))
        s = d_2c * s1 / (d_1c + d_2c) + d_1c * s2 / (d_1c + d_2c)
        # print(closest_points)
        # print((self.x_eval(s1), self.y_eval(s1)), (x1, y1))
        return s
    
    def frenet_to_global(self, s: float, ey: float):
        x = self.x_eval(s)
        y = self.y_eval(s)
        x_d = self.x_eval_d(s)
        y_d = self.y_eval_d(s)
        
        
        return
    
    def track_to_global(self, e_y, e_yaw, s):
        # line 557: track_to_global
        dx_ds = self.x_eval_d(s)
        dy_ds = self.y_eval_d(s)

        proj = np.array([self.x_eval(s), self.y_eval(s)])
        pos = proj + normalize_vector(np.array([-dy_ds, dx_ds])) * e_y
        yaw = e_yaw + np.arctan2(dy_ds, dx_ds) # e_yaw = yaw - yaw_des
        return np.array([pos[0], pos[1], yaw])
    
    def global_to_track(self, x, y, yaw, s):
        # line 543: global_to_track
        x_proj = self.x_eval(s)
        y_proj = self.y_eval(s)
        e_y = np.sqrt((x - x_proj)**2 + (y - y_proj)**2)
        dx_ds = self.x_eval_d(s)
        dy_ds = self.y_eval_d(s)
        if dx_ds * (y - y_proj) - dy_ds * (x - x_proj) > 0:
            e_y = -e_y
        e_yaw = yaw - np.arctan2(dy_ds, dx_ds)
        while e_yaw > np.pi:
            e_yaw -= 2*np.pi
        while e_yaw < -np.pi:
            e_yaw += 2*np.pi
        return -e_y, e_yaw, s
    
    def update_half_width(self, thetas: np.ndarray):
        """
        Update left right TODO
        """
        widths = np.ones_like(thetas, dtype=float) * 1.2
        return widths
    
    def get_left_half_width(self, theta: float) -> float:
        """
        TODO: Update left half width
        """
        idx = max(0, min(len(self.centerline_points) - 1, int(np.floor(theta / self.step))))
        return self.centerline_points[idx, 2]
    
    def get_right_half_width(self, theta: float) -> float:
        idx = max(0, min(len(self.centerline_points) - 1, int(np.floor(theta / self.step))))
        return self.centerline_points[idx, 3]
    
    def set_half_width(self, theta: float, left: float, right: float):
        idx = max(0, min(len(self.centerline_points) - 1, int(np.floor(theta / self.step))))
        self.centerline_points[idx, 2] = left
        self.centerline_points[idx, 3] = right

    def get_centerline_points_curvature(self, theta: float) -> float:
        dx_dtheta = self.x_eval_d(theta)
        dy_dtheta = self.y_eval_d(theta)
        dx_ddtheta = self.x_eval_dd(theta)
        dy_ddtheta = self.y_eval_dd(theta)
        return (dx_dtheta * dy_ddtheta - dy_dtheta * dx_ddtheta) / (dx_dtheta**2 + dy_dtheta**2)**1.5

    def get_centerline_points_radius(self, theta: float) -> float:
        return 1.0 / self.get_centerline_points_curvature(theta)
    
    def wrap_theta(self, theta: float) -> float:
        while (theta > self.length):
            theta -= self.length
        while (theta < 0):
            theta += self.length
        return theta
    
    def wrap_angle(self, angle):
        """
        Wrap angle to [-pi, pi]
        """
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle
    
    def diff_angle(self, angle1: float, angle2: float) -> float:
        """
        Compute difference between angle1 and angle 2 (angle1 - angle2)
        """
        diff = self.wrap_angle(angle1) - self.wrap_angle(angle2)
        while abs(diff) > np.pi:
            if diff > 0:
                diff = diff - 2 * np.pi
            else:
                diff = diff + 2 * np.pi
        return diff

    ## Spline evaluation
    def x_eval(self, theta: float) -> float:
        theta = self.wrap_theta(theta)
        return self.x_spline(theta)
    
    def y_eval(self, theta: float) -> float:
        theta = self.wrap_theta(theta)
        return self.y_spline(theta)
    
    def x_eval_d(self, theta: float) -> float:
        theta = self.wrap_theta(theta)
        return self.x_spline.eval_d(theta)
    
    def y_eval_d(self, theta: float) -> float:
        theta = self.wrap_theta(theta)
        return self.y_spline.eval_d(theta)
    
    def x_eval_dd(self, theta: float) -> float:
        theta = self.wrap_theta(theta)
        return self.x_spline.eval_dd(theta)
    
    def y_eval_dd(self, theta: float) -> float:
        theta = self.wrap_theta(theta)
        return self.y_spline.eval_dd(theta)
    
    def get_phi(self, theta: float) -> float:
        theta = self.wrap_theta(theta)
        dx_dtheta = self.x_eval_d(theta)
        dy_dtheta = self.y_eval_d(theta)
        return np.arctan2(dy_dtheta, dx_dtheta)
    
    def get_cum_distance(self, xy:np.ndarray):
        """
        Get the cumulative distance of the track at a given xy of shape (N,)
        For computing s
        """
        dis = np.zeros(xy.shape[0])
        dis[1:] = np.linalg.norm(xy[1:, :] - xy[:-1, :], axis=1) # (n-1)
        return np.cumsum(dis) #(n, )


def normalize_vector(vec):
    norm = np.linalg.norm(vec)
    return vec / norm
    
if __name__ == "__main__":
    a, b = np.array([1, 1])
    b = np.array([1, -1])
    print(np.cross(a, b))
    print(a)
    pass