#!/usr/bin/env python3
import numpy as np
from dataclasses import dataclass
from spline import Spline

# @dataclass
# class Waypoint:
#     x: float
#     y: float
#     theta: float
#     left: float
#     right: float


class Track:
    def __init__(self, centerline_points: str):
        self.centerline_points = self.load_waypoints(centerline_points) # (N, 5) [x, y, left, right, theta]
        self.centerline_xy = self.centerline_points[:, :2]
        self.x_spline: Spline = None
        self.y_spline: Spline = None
        self.step = 0.05 # step size
        self.length = 0.0

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
        new_points = np.zeros((N+1, 5))
        _, starting_index = self.find_closest_waypoint(x, y)
        new_points[:N, :4] = np.roll(self.centerline_points[:, :4], -starting_index, axis=0)
        new_points[N, :4] = new_points[0, :4] # CLOSE THE LOOP
        ## Find s for each point
        new_points[:, 4] = self.get_cum_distance(new_points[:, :2])
        self.centerline_points = new_points
        self.centerline_xy = self.centerline_points[:, :2]
        self.length = self.centerline_points[-1, 4]
        ## Fit spline y = f(s), x = f(s)
        self.x_spline = Spline(self.centerline_points[:, 4], self.centerline_points[:, 0])
        self.y_spline = Spline(self.centerline_points[:, 4], self.centerline_points[:, 1])
        if refine:
            self.refine_uniform_waypoint()
        np.savetxt("map/refined_centerline.csv", self.centerline_points, delimiter=",")
        return
    
    def get_cum_distance(self, xy:np.ndarray):
        """
        Get the cumulative distance of the track at a given xy of shape (N, 2)
        """
        dis = np.zeros(xy.shape[0])
        dis[1:] = np.linalg.norm(xy[1:, :] - xy[:-1, :], axis=1) # (n-1)
        return np.cumsum(dis) #(n, )
    
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
        x = np.append(x, x[0]) # close the loop
        y = np.append(y, y[0]) # close the loop
        xy = np.hstack([x[:, None], y[:, None]])
        s_new = self.get_cum_distance(xy)
        left_right_distance = self.find_width(x)
        new_points = np.hstack([xy, left_right_distance[:], s_new[:, None]])
        self.centerline_points = new_points
        self.centerline_xy = self.centerline_points[:, :2]
        self.length = self.centerline_points[-1, 4]

    def load_waypoints(self, csv_path: str):
        self.waypoints = csv_path
        with open(csv_path, "r") as f:
            waypoint = np.loadtxt(f, delimiter=",") # [[x, y, left, right], ...]
        return waypoint

    def find_width(self, xy: np.ndarray):
        """
        Update left right TODO
        """
        widths = np.zeros_like(xy)
        return widths

    def find_closest_waypoint(self, x: float, y: float, n: int = 1):
        """
        Return closest n waypoints to the given x, y position
        """
        dist = np.linalg.norm(self.centerline_xy - np.array([x, y]), axis=1)
        if n == 1:
            ind = np.argmin(dist)
        else:
            dist_sorted = np.argsort(dist)
            ind = dist_sorted[:np.min(n, len(dist_sorted))]
        return self.centerline_points[ind].reshape(n, -1), ind # (n, 5) [x, y, theta, left, right]

    def get_theta(self, x: float, y: float) -> float:
        """
        Given a position, estimate progress on track
        """
        closest_points, _ = self.find_closest_waypoint(x, y)
        # print(closest_points)
        return closest_points[0, 4]
    
    def find_theta(self, point: np.ndarray) -> float:
        dist = np.linalg.norm(self.centerline_points[:, :2] - point, axis=1)
        ind = np.argmin(dist)
        return ind * self.step
    
    def wrap_theta(self, theta: float) -> float:
        while (theta > self.length):
            theta -= self.length
        while (theta < 0):
            theta += self.length
        return theta

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
    
    def get_left_half_width(self, theta: float) -> float:
        idx = max(0, min(len(self.centerline_points) - 1, np.floor(theta / self.step)))
        return self.centerline_points[idx, 2]
    
    def get_right_half_width(self, theta: float) -> float:
        idx = max(0, min(len(self.centerline_points) - 1, np.floor(theta / self.step)))
        return self.centerline_points[idx, 3]
    
    def set_half_width(self, theta: float, left: float, right: float):
        idx = max(0, min(len(self.centerline_points) - 1, np.floor(theta / self.step)))
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
        
if __name__ == "__main__":
    a = np.arange(0, 1, 1/3)
    print(a)
    pass