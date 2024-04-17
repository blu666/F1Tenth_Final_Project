import numpy as np
from dataclasses import dataclass


@dataclass
class Waypoint:
    x: float
    y: float
    theta: float
    left: float
    right: float


class Track:
    def __init__(self, centerline_points: str):
        self.centerline_points = self.load_waypoints(centerline_points) # (N, 5) [x, y, theta, left, right], should form a loop
        self.centerline_xy = self.centerline_points[:, :2]
        self.x_spline = None
        self.y_spline = None
        self.step = 0.05 # step size
        self.length = None

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

if __name__ == "__main__":
    a = np.array([[1, 1, 1, 1, 1], [4, 4, 4, 4, 4], [3, 3, 4, 4, 4]])
    dist = np.array([[1, 0], [3, 0], [2, 0]])
    dist = np.linalg.norm(a, axis=1)
    s = np.argsort(dist)
    
    print(a[s[:2]])
    pass