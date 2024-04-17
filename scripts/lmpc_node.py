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

VEL_THRESH = 0.8

# class def for RRT
class LMPC(Node):
    def __init__(self):
        super().__init__('lmpc_node')
        self.car = CarParams()
        self.is_first_run = True
        self.create_subscription(Odometry, 'ego_racecar/odom', self.pose_callback, 10)
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

    # def get_linearized_dynamics(self, x, u, use_dyn):
    #     # line 566: get_linearized_dynamics
    #     pass
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
    print("RRT Initialized")
    lmpc_node = LMPC()
    rclpy.spin(lmpc_node)

    lmpc_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()