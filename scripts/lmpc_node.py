#!/usr/bin/env python3
import numpy as np
import math

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Point
from ackermann_msgs.msg import AckermannDriveStamped, AckermannDrive
from nav_msgs.msg import OccupancyGrid, Odometry
from typing import List

import csv
from dataclasses import dataclass
from matplotlib import pyplot as plt
from copy import deepcopy
import scipy
from scipy.spatial.transform import Rotation as R
import tf2_ros
from visualization_msgs.msg import Marker, MarkerArray
from rclpy.parameter import Parameter, ParameterType
from sensor_msgs.msg import PointCloud
from car_params import CarParams, load_default_car_params
from track import Track ## TODO: implement Track class
from scipy import sparse
import cvxpy as cp
from osqp import OSQP


# class def for RRT
class LMPC(Node):
    def __init__(self):
        super().__init__('lmpc_node')
        #==== Load Track
        self.Track = Track("map/refined_centerline.csv", initialized=True)

        #==== global variables
        self.s_prev = 0
        self.s_curr = 0
        self.time = 0
        self.iter = 2
        self.car_pos = np.zeros(2)
        self.yaw = 0
        self.vel = 0
        self.yawdot = 0
        self.slip_angle = 0

        self.curr_traj = []
        self.QPSol = None
        self.terminal_state_pred = None
        
        self.map_to_car_rotation = None
        self.map_to_car_translation = None
        
        self.first_run = True
        self.SS = None
        self.use_dynamics = False
        
        #==== Params: TODO: set_params
        self.car: CarParams = load_default_car_params()
        self.nx = 6 # dim of state space [x, y, yaw, v, omega, slip]
        self.nu = 2 # dim of control space
        self.ts = 0.05 # time steps

        #==== Load SS from data
        self.init_SS("map/inital_ss.csv")

        #==== Create pub sub
        self.create_subscription(Odometry, 'ego_racecar/odom', self.odom_callback, 10)
        self.drive_publisher = self.create_publisher(AckermannDriveStamped, '/drive', 10)
        self.create_timer(1 / 20.0, self.lmpc_run)
        self.get_logger().info("LMPC Node Initialized")


    def lmpc_run(self):
        pass
    

    def odom_callback(self, pose_msg: Odometry):
        #==== Update car state
        current_pose = np.array([pose_msg.pose.pose.position.x,
                                 pose_msg.pose.pose.position.y])
        current_heading = R.from_quat([pose_msg.pose.pose.orientation.x,
                                       pose_msg.pose.pose.orientation.y,
                                       pose_msg.pose.pose.orientation.z,
                                       pose_msg.pose.pose.orientation.w])
        self.map_to_car_translation = np.array([pose_msg.pose.pose.position.x,
                                                pose_msg.pose.pose.position.y,
                                                pose_msg.pose.pose.position.z])
        self.map_to_car_rotation = R.from_quat([pose_msg.pose.pose.orientation.x,
                                                pose_msg.pose.pose.orientation.y,
                                                pose_msg.pose.pose.orientation.z,
                                                pose_msg.pose.pose.orientation.w])
        self.yaw = current_heading.as_euler('zyx')[0]
        self.vel = np.linalg.norm([pose_msg.twist.twist.linear.x, pose_msg.twist.twist.linear.y])
        self.yawdot = pose_msg.twist.twist.angular.z
        self.slip_angle = np.arctan2(pose_msg.twist.twist.linear.y, pose_msg.twist.twist.linear.x)
        #==== Update progress on track
        self.s_curr = self.Track.find_theta(current_pose)
        
        
        # if (not self.use_dynamics) and (self.vel > self.car.DYNA_VEL_THRESH):
        #     self.use_dynamics = True
        # elif(self.use_dynamics) and (self.vel < self.car.DYNA_VEL_THRESH):
        #     self.use_dynamics = False
        # if (vel > 4.5):

        # self.lmpc_run()

    def init_SS(self, data_file: str):
        pass

    def select_terminal_candidate(self):
        # line 456: select_terminal_candidate
        pass



    def select_convex_ss(self, iter_start, iter_end, s):
        # line 477: select_convex_safe_set
        pass


    def find_nearest_point(self, trajectory, s):
        # line 524: find_nearest_point
        # print(s)
        low, high = 0, len(trajectory)
        while low <= high:
            mid = (low + high) // 2
            if trajectory[mid].s == s:
                return mid
            elif trajectory[mid].s < s:
                low = mid + 1
            else:
                high = mid - 1

        if abs(trajectory[low].s - s) < abs(trajectory[high].s - s):
            return low
        else:
            return high
        

    def update_cost_to_go(self, trajectory):
        return 

    def track_to_global(self, e_y, e_yaw, s):
        # line 557: track_to_global
        dx_ds = self.Track.x_eval_d(s)
        dy_ds = self.Track.y_eval_d(s)

        proj = np.array([self.Track.x_eval(s), self.Track.y_eval(s)])
        pos = proj + normalize_vector(np.array([-dy_ds, dx_ds])) * e_y
        yaw = e_yaw + np.arctan2(dy_ds, dx_ds)
        return np.array([pos[0], pos[1], yaw])

    def global_to_track(self, x, y, yaw, s):
        # line 543: global_to_track
        x_proj = self.Track.x_eval(s)
        y_proj = self.Track.y_eval(s)
        e_y = np.sqrt((x - x_proj)**2 + (y - y_proj)**2)
        dx_ds = self.Track.x_eval_d(s)
        dy_ds = self.Track.y_eval_d(s)
        if dx_ds * (y - y_proj) - dy_ds * (x - x_proj) > 0:
            e_y = -e_y
        e_yaw = yaw - np.arctan2(dy_ds, dx_ds)
        while e_yaw > np.pi:
            e_yaw -= 2*np.pi
        while e_yaw < -np.pi:
            e_yaw += 2*np.pi
        return np.array([e_y, e_yaw, s])

    def get_linearized_dynamics(self, Ad: np.ndarray,
                                Bd: np.ndarray,
                                hd: np.ndarray,
                                x_op: np.ndarray,
                                u_op: np.ndarray,
                                use_dynamics: bool):
        """
        line 566: get_linearized_dynamics
        by Henry
        """
        return

    
        
    def wrap_angle(self, angle, ref_angle):
        # line 688: wrap_angle
        while angle - ref_angle > np.pi:
            angle -= 2*np.pi
        while angle - ref_angle < -np.pi:
            angle += 2*np.pi
        return angle

    def solve_MPC(self, terminal_candidate):
        # line 693: solve_MPC
        return


        

    def apply_control(self):
        # line 938: apply_control
        accel = ... # TODO: implement
        steer = ...

        self.get_logger().info(f"accel_cmd: {accel}, steer_cmd: {steer}, slip_angle: {self.slip_angle}")
        steer = np.clip(steer, -self.car.STEER_MAX, self.car.STEER_MAX)

        drive_msg = AckermannDriveStamped()
        drive_msg.header.stamp = self.get_clock().now().to_msg()
        drive_msg.header.frame_id = 'base_link'
        drive_msg.drive.steering_angle = steer
        drive_msg.drive.steering_angle_velocity = 1.0
        drive_msg.drive.speed = self.vel + accel * (1.0/20.0)
        drive_msg.drive.acceleration = accel
        self.drive_publisher.publish(drive_msg)

def normalize_vector(vec):
    norm = np.linalg.norm(vec)
    return vec / norm

def main(args=None):
    rclpy.init(args=args)
    # print("RRT Initialized")
    lmpc_node = LMPC()
    rclpy.spin(lmpc_node)

    lmpc_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()