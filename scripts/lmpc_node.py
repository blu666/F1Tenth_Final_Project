#!/usr/bin/env python3
"""
This file contains the class definition for tree nodes and RRT
Before you start, please read: https://arxiv.org/pdf/1105.1186.pdf
"""
import numpy as np
from numpy import linalg as LA
import math

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
# from geometry_msgs.msg import PoseStamped
# from geometry_msgs.msg import PointStamped
# from geometry_msgs.msg import Pose
from geometry_msgs.msg import Point
# from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped, AckermannDrive
from nav_msgs.msg import OccupancyGrid, Odometry

from dataclasses import dataclass
from scipy.spatial.transform import Rotation as R
import tf2_ros
from visualization_msgs.msg import Marker, MarkerArray
from rclpy.parameter import Parameter, ParameterType
from sensor_msgs.msg import PointCloud
from occupancy_grid import CarOccupancyGrid
import track

nx = 6 # dim of state space
nu = 2 # dim of control space

@dataclass
class Sample:
    x: np.ndarray
    u: np.ndarray
    s: float
    time: int
    iter: int
    cost: int

VEL_THRESH = 0.8

# class def for RRT
class LMPC(Node):
    def __init__(self):
        super().__init__('lmpc_node')
        
        self.create_subscription(Odometry, 'ego_racecar/odom', self.odom_callback, 10)
        self.drive_publisher = self.create_publisher(AckermannDriveStamped, '/drive', 10)

        self.map_to_car_rotation = None
        self.map_to_car_translation = None
        
        self.first_run = True
        self.SS = None
        self.use_dynamics = True

        self.s_prev = 0
        self.s_curr = 0
        self.time = 0

    def lmpc_run(self):
        # line 387: run
        if self.first_run:
            # reset QP solution
            pass

        # check if new lap
        ...

        # select terminal candidate
        terminal_candidate = self.select_terminal_candidate()
        self.solve_MPC(terminal_candidate)
        self.apply_control()
        self.add_point()

        terminal_state_pred = ...
        self.s_prev = self.s_curr
        self.time += 1
        self.first_run = False
        

    def odom_callback(self, pose_msg):
        current_pose = np.array([pose_msg.pose.pose.position.x, pose_msg.pose.pose.position.y])
        current_heading = R.from_quat([pose_msg.pose.pose.orientation.x, pose_msg.pose.pose.orientation.y, pose_msg.pose.pose.orientation.z, pose_msg.pose.pose.orientation.w])
        self.map_to_car_translation = np.array([pose_msg.pose.pose.position.x, pose_msg.pose.pose.position.y, pose_msg.pose.pose.position.z])
        self.map_to_car_rotation = R.from_quat([pose_msg.pose.pose.orientation.x, pose_msg.pose.pose.orientation.y, pose_msg.pose.pose.orientation.z, pose_msg.pose.pose.orientation.w])

        s_curr = track.findTheta(current_pose[0], current_pose[1], 0, True)
        yaw = ...
        vel = ...
        yawdot = ...
        slip_angle = ...

        if (not self.use_dynamics) and (vel > VEL_THRESH):
            self.use_dynamics = True
        elif(self.use_dynamics) and (vel < VEL_THRESH):
            self.use_dynamics = False
        # if (vel > 4.5):

        self.lmpc_run()



    def init_SS(self, data_file):
        # line 244: init_SS_from_data
        pass

    def select_terminal_candidate(self):
        # line 456: select_terminal_candidate
        pass
    
    def add_point(self):
        # line 465: add_point
        pass

    def select_convex_ss(self, iter_start, iter_end, s):
        # line 477: select_convex_safe_set
        pass
    
    def find_nearest_point(self, trajectory, s):
        # line 524: find_nearest_point
        pass

    def update_cost_to_go(self, trajectory):
        # line 536: update_cost_to_go
        pass

    def track_to_global(self, e_y, e_yaw, s):
        # line 557: track_to_global
        pass

    def global_to_track(self, x, y, yaw, s):
        # line 543: global_to_track
        pass

    def get_linearized_dynamics(self, x, u, use_dyn):
        # line 566: get_linearized_dynamics
        pass

    def wrap_angle(self, angle, ref_angle):
        # line 688: wrap_angle
        pass

    def solve_MPC(self, terminal_candidate):
        # line 693: solve_MPC
        pass

    def apply_control(self):
        # line 938: apply_control
        pass


def main(args=None):
    rclpy.init(args=args)
    print("LMPC Initialized")
    lmpc_node = LMPC()
    rclpy.spin(lmpc_node)

    lmpc_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()