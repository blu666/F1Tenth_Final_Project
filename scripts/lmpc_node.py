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

# TODO: import as you need

import csv
from dataclasses import dataclass
from matplotlib import pyplot as plt
from copy import deepcopy
from scipy.spatial.transform import Rotation as R
import tf2_ros
from visualization_msgs.msg import Marker, MarkerArray
from rclpy.parameter import Parameter, ParameterType
from sensor_msgs.msg import PointCloud
from car_params import CarParams
# from rrt_star import RRT_Star

nx = 6 # dim of state space [x, y, yaw, v, omega, slip]
nu = 2 # dim of control space

@dataclass
class SS_Sample:
    x: np.ndarray
    u: np.ndarray
    s: float
    time: int
    iter: int
    cost: int

# class def for RRT
class LMPC(Node):
    def __init__(self):
        super().__init__('lmpc_node')
        self.is_first_run = True
        self.car = CarParams()

    
    def get_linearized_dynamics(self, Ad: np.ndarray,
                                Bd: np.ndarray,
                                hd: np.ndarray,
                                x_op: np.ndarray,
                                u_op: np.ndarray,
                                use_dynamics: bool):
        """
        
        """
        yaw = x_op[2]
        v = x_op[3]
        accel = u_op[0]
        steer = u_op[1]
        yaw_dot = x_op[4]
        slip_angle = x_op[5]
        
        dynamics = np.zeros(6)
        h = np.zeros(6)
        
        if use_dynamics:
            g = 9.81
            rear_val = self.car.l_r - accel * self.car.h_cg
            front_val = self.car.l_f + accel * self.car.h_cg

        
        self.create_subscription(Odometry, 'ego_racecar/odom', self.pose_callback, 10)
        self.drive_publisher = self.create_publisher(AckermannDriveStamped, '/drive', 10)

        self.map_to_car_rotation = None
        self.map_to_car_translation = None

        return A, B, h

    def pose_callback(self, msg):
        current_pose = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y])
        current_heading = R.from_quat([msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w])
        


    

def main(args=None):
    rclpy.init(args=args)
    print("RRT Initialized")
    lmpc_node = LMPC()
    rclpy.spin(lmpc_node)

    lmpc_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()