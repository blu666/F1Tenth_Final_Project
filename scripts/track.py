import numpy as np
from dataclasses import dataclass


# @dataclass
# class Waypoint:
#     x: float
#     y: float
#     theta: float
#     left: float
#     right: float


class Track:
    def __init__(self, centerline_points: str):
        self.centerline_points = self.load_waypoints(centerline_points) # (N, 5) [x, y, theta, left, right], should form a loop
        self.centerline_xy = self.centerline_points[:, :2]
        self.x_spline = None
        self.y_spline = None
        self.step = 0.05 # step size
        self.length = 0.0

    def refine_centerline(self):
        """
        downsample centerline waypoints, fit spline, sample new set of points.
        """
        return

    def load_waypoints(self, csv_path: str):
        self.waypoints = csv_path
        with open(csv_path, "r") as f:
            waypoint = np.loadtxt(f, delimiter=",")
            # [[x, y, theta, left, right], ...] -> [Waypoint(x, y, theta, left, right), ...]
            # TODO: [[x, y]], and refine.
        return waypoint

    def initialize_width(self):
        pass


    def find_closest_waypoint(self, x: float, y: float, n: int = 1):
        """
        Return closest n waypoints to the given x, y position
        """
        # min_dist = float('inf')
        # closest_point = None
        # for point in self.centerline_points:
        #     dist = np.sqrt((x - point[0])**2 + (y - point[1])**2)
        #     if dist < min_dist:
        #         min_dist = dist
        #         closest_point = point
        dist = np.linalg.norm(self.centerline_xy - np.array([x, y]), axis=1)
        if n == 1:
            dist_min = np.argmin(dist)
            closest_points = self.centerline_points[dist_min]
        else:
            dist_sorted = np.argsort(dist)
            closest_points = self.centerline_points[dist_sorted[:np.min(n, len(dist_sorted))]]
        return closest_points # (n, 5) [x, y, theta, left, right]

    def get_theta(self, x: float, y: float) -> float:
        """
        Given a position, estimate progress on track
        """
        closest_points = self.find_closest_waypoint(x, y)
        return closest_points[0, 2]
    
    def wrap_theta(self, theta: float) -> float:
        while (theta > self.length):
            theta -= self.length
        while (theta < 0):
            theta += self.length

    def x_eval(self, theta: float) -> float:
        self.wrap_theta(theta)
        return self.x_spline(theta)
    
    def y_eval(self, theta: float) -> float:
        self.wrap_theta(theta)
        return self.y_spline(theta)
    
    def x_eval_d(self, theta: float) -> float:
        self.wrap_theta(theta)
        return self.x_spline.eval_d(theta)
    
    def y_eval_d(self, theta: float) -> float:
        self.wrap_theta(theta)
        return self.y_spline.eval_d(theta)
    
    def x_eval_dd(self, theta: float) -> float:
        self.wrap_theta(theta)
        return self.x_spline.eval_dd(theta)
    
    def y_eval_dd(self, theta: float) -> float:
        self.wrap_theta(theta)
        return self.y_spline.eval_dd(theta)
    
    def get_phi(self, theta: float) -> float:
        self.wrap_theta(theta)
        dx_dtheta = self.x_eval_d(theta)
        dy_dtheta = self.y_eval_d(theta)
        return np.arctan2(dy_dtheta, dx_dtheta)
    
    def get_left_half_width(self, theta: float) -> float:
        idx = max(0, min(len(self.centerline_points) - 1, np.floor(theta / self.step)))
        return self.centerline_points[idx, ...] # left half width (3)?
    
    def get_right_half_width(self, theta: float) -> float:
        idx = max(0, min(len(self.centerline_points) - 1, np.floor(theta / self.step)))
        return self.centerline_points[idx, ...]
    
    def set_half_width(self, theta: float, left: float, right: float):
        idx = max(0, min(len(self.centerline_points) - 1, np.floor(theta / self.step)))
        ##########
        # centerline columns
        ##########
        self.centerline_points[idx, 3] = left
        self.centerline_points[idx, 4] = right

    def get_centerline_points_curvature(self, theta: float) -> float:
        dx_dtheta = self.x_eval_d(theta)
        dy_dtheta = self.y_eval_d(theta)
        dx_ddtheta = self.x_eval_dd(theta)
        dy_ddtheta = self.y_eval_dd(theta)
        return (dx_dtheta * dy_ddtheta - dy_dtheta * dx_ddtheta) / (dx_dtheta**2 + dy_dtheta**2)**1.5

    def get_centerline_points_radius(self, theta: float) -> float:
        return 1.0 / self.get_centerline_points_curvature(theta)
        
if __name__ == "__main__":
    a = np.array([[1, 1, 1, 1, 1], [4, 4, 4, 4, 4], [3, 3, 4, 4, 4]])
    dist = np.array([[1, 0], [3, 0], [2, 0]])
    dist = np.linalg.norm(a, axis=1)
    s = np.argsort(dist)
    
    print(a[s[:2]])
    pass