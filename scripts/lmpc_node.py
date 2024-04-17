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
        A = np.zeros((nx, nx))
        B = np.zeros((nx, nu))
        if use_dynamics:
            g = 9.81
            rear_val = g * self.car.l_r - accel * self.car.h_cg
            front_val = g * self.car.l_f + accel * self.car.h_cg

            dynamics[0] = v * np.cos(yaw+slip_angle)
            dynamics[1] = v * np.sin(yaw+slip_angle)
            dynamics[2] = yaw_dot
            dynamics[3] = accel
            dynamics[4] = (self.car.friction_coeff * self.car.mass / (self.car.I_z * self.car.wheelbase)) \
                * (self.car.l_f * self.car.cs_f * steer * rear_val + \
                    slip_angle * (self.car.l_r * self.car.cs_r * front_val - self.car.l_f * self.car.cs_f * rear_val) \
                        - yaw_dot/v * (math.pow(self.car.l_f, 2) * self.car.cs_f * rear_val\
                            + math.pow(self.car.l_r, 2) * self.car.cs_r * front_val))
            dynamics[5] = (self.car.friction_coeff / (v * (self.car.l_r + self.car.l_f))) \
                * (self.car.cs_f * steer * rear_val - slip_angle * (self.car.cs_r * front_val + self.car.cs_f * rear_val) \
                    + (yaw_dot/v) * (self.car.cs_r * self.car.l_r * front_val - self.car.cs_f * self.car.l_f * rear_val)) \
                        - yaw_dot
            
            dfyawdot_dv = (self.car.friction_coeff * self.car.mass / (self.car.I_z * self.car.wheelbase))\
                * (math.pow(self.car.l_f, 2) * self.car.cs_f * (rear_val) + math.pow(self.car.l_r, 2) * self.car.cs_r * (front_val))\
                * yaw_dot / math.pow(v, 2)

            dfyawdot_dyawdot = -(self.car.friction_coeff * self.car.mass / (self.car.I_z * self.car.wheelbase))\
                            * (math.pow(self.car.l_f, 2) * self.car.cs_f * (rear_val) + math.pow(self.car.l_r, 2) * self.car.cs_r * (front_val))/v

            dfyawdot_dslip = (self.car.friction_coeff * self.car.mass / (self.car.I_z * self.car.wheelbase))\
                                * (self.car.l_r * self.car.cs_r * (front_val) - self.car.l_f * self.car.cs_f * (rear_val))

            dfslip_dv = -(self.car.friction_coeff / (self.car.l_r + self.car.l_f)) *\
                        (self.car.cs_f * steer * rear_val - slip_angle * (self.car.cs_r * front_val + self.car.cs_f * rear_val))/math.pow(v,2)\
                    -2*(self.car.friction_coeff / (self.car.l_r + self.car.l_f)) * (self.car.cs_r * self.car.l_r * front_val - self.car.cs_f * self.car.l_f * rear_val) * yaw_dot/math.pow(v,3)

            dfslip_dyawdot = (self.car.friction_coeff / (math.pow(v,2) * (self.car.l_r + self.car.l_f))) * (self.car.cs_r * self.car.l_r * front_val - self.car.cs_f * self.car.l_f * rear_val) - 1

            dfslip_dslip = -(self.car.friction_coeff / (v * (self.car.l_r + self.car.l_f)))*(self.car.cs_r * front_val + self.car.cs_f * rear_val)

            dfyawdot_da = (self.car.friction_coeff * self.car.mass / (self.car.I_z * self.car.wheelbase))\
                    *(-self.car.l_f*self.car.cs_f*self.car.h_cg*steer + self.car.l_r*self.car.cs_r*self.car.h_cg*slip_angle + self.car.l_f*self.car.cs_f*self.car.h_cg*slip_angle\
                    - (yaw_dot/v)*(-math.pow(self.car.l_f,2)*self.car.cs_f*self.car.h_cg) + math.pow(self.car.l_r,2)*self.car.cs_r*self.car.h_cg)

            dfyawdot_dsteer = (self.car.friction_coeff * self.car.mass / (self.car.I_z * self.car.wheelbase)) *\
                        (self.car.l_f * self.car.cs_f * rear_val)

            dfslip_da = (self.car.friction_coeff / (v * (self.car.l_r + self.car.l_f))) *\
                    (-self.car.cs_f*self.car.h_cg*steer - (self.car.cs_r*self.car.h_cg - self.car.cs_f*self.car.h_cg)*slip_angle +\
                    (self.car.cs_r*self.car.h_cg*self.car.l_r + self.car.cs_f*self.car.h_cg*self.car.l_f)*(yaw_dot/v))

            dfslip_dsteer = (self.car.friction_coeff / (v * (self.car.l_r + self.car.l_f))) *\
                    (self.car.cs_f * rear_val)
            
            A[0, 2] = -v * np.sin(yaw+slip_angle)
            A[0, 3] = np.cos(yaw+slip_angle)
            A[0, 5] = -v * np.sin(yaw+slip_angle)
            A[1, 2] = v * np.cos(yaw+slip_angle)
            A[1, 3] = np.sin(yaw+slip_angle)
            A[1, 5] = v * np.cos(yaw+slip_angle)
            A[2, 4] = 1
            A[4, 3] = dfyawdot_dv
            A[4, 4] = dfyawdot_dyawdot
            A[4, 5] = dfyawdot_dslip
            A[5, 3] = dfslip_dv
            A[5, 4] = dfslip_dyawdot
            A[5, 5] = dfslip_dslip
            
            B[3, 0] = 1
            B[4, 0] = dfyawdot_da
            B[4, 1] = dfyawdot_dsteer
            B[5, 0] = dfslip_da
            B[5, 1] = dfslip_dsteer
        else:
            dynamics[0] = v * np.cos(yaw)
            dynamics[1] = v * np.sin(yaw)
            dynamics[2] = v * np.tan(steer) / self.car.wheelbase
            dynamics[3] = accel
            dynamics[4] = 0
            dynamics[5] = 0

            A[0, 2] = -v * np.sin(yaw)
            A[0, 3] = np.cos(yaw)
            A[1, 2] = v * np.cos(yaw)
            A[1, 3] = np.sin(yaw)
            A[2, 3] = np.tan(steer) / self.car.wheelbase
            
            B[2, 1] = v / (self.car.wheelbase * np.cos(steer)**2)
            B[3, 0] = 1
        return Ad, Bd, hd

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