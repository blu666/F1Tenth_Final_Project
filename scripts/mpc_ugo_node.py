#!/usr/bin/env python3
import sys
sys.path.append("./") # Run under ./lmpc
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
from utils.params import CarParams, load_default_car_params
from utils.track import Track
from utils.utilities import load_init_ss
from utils.initControllerParameters import initLMPCParams, initMPCParams
from utils.PredictiveModel import PredictiveModel
from utils.utilities import Regression
from scipy import sparse
import cvxpy as cp
from osqp import OSQP
from utils.PredictiveControllers import MPC

class MPC(Node):
    def __init__(self):
        super().__init__('mpc_node')
        # create ROS subscribers and publishers
        #       use the MPC as a tracker (similar to pure pursuit)
        self.drive_pub = self.create_publisher(AckermannDriveStamped, '/drive', 10)
        self.pose_sub = self.create_subscription(Odometry, '/ego_racecar/odom', self.pose_callback, 1)
        
        self.waypoints_publisher = self.create_publisher(MarkerArray, '/pure_pursuit/waypoints', 10)
        # self.goalpoint_publisher = self.create_publisher(Marker, '/pure_pursuit/goalpoint', 5)
        # self.testpoint_publisher = self.create_publisher(MarkerArray, '/pure_pursuit/testpoints', 10)
        # self.refpoint_publisher = self.create_publisher(MarkerArray, '/pure_pursuit/refpoints', 10)

        
        # self.waypoints = self.load_waypoints("./map/levine_mpc/levine_line_v_yaw.csv")
        # self.publish_waypoints()

        # self.config = mpc_config()
        # self.odelta_v = None
        # self.odelta = None
        # self.oa = None
        # self.init_flag = 0
        self.pose_msg = None
        self.x = np.zeros(6)
        self.vt = 0.5
        self.dt = 0.1
        self.first_run = True
        self.first_mpc_run = True
        self.mode = 0 # 0: PID, 1: MPC
        self.lap = 0
        self.pid_laps = 1

        self.xpid = None
        self.upid = None
        self.flag = False


        # self.create_timer(1./30., self.run_mpc)
        # initialize MPC problem
        # self.mpc_prob_init()

        self.car: CarParams = load_default_car_params()
        self.pid_init(1.0)
        
        self.create_timer(1./30., self.run_pid)
        self.get_logger().info("MPC Node Initialized")

        

    
    def pose_callback(self, pose_msg):
        self.pose_msg = pose_msg
        if self.first_run:
            #==== Initialize Track & reset starting point to spawn point
            self.Track = Track("./map/levine/centerline.csv")
            self.Track.reset_starting_point(pose_msg.pose.pose.position.x,
                                            pose_msg.pose.pose.position.y,
                                            refine=True)
            self.publish_waypoints(self.Track.centerline_xy)
            # self.get_logger().info("@=>Init: Track Loaded")
            # #==== Initialize LMPC
            # self.initialize_lmpc(self.N, self.n, self.d, self.Track)
            # self.get_logger().info("@=>Init: LMPC Initialized")
            self.first_run = False
        x = np.zeros(6)
        vx_global = pose_msg.twist.twist.linear.x
        vy_global = pose_msg.twist.twist.linear.y
        wz = pose_msg.twist.twist.angular.z
        yaw = R.from_quat([pose_msg.pose.pose.orientation.x, pose_msg.pose.pose.orientation.y, pose_msg.pose.pose.orientation.z, pose_msg.pose.pose.orientation.w]).as_euler('zyx')[0]
        states = self.Track.get_states(pose_msg.pose.pose.position.x, pose_msg.pose.pose.position.y, yaw)

        x[0] = vx_global # vx_global * np.cos(yaw) + vy_global * np.sin(yaw)
        x[1] = np.random.randn(1) * 1e-5 #-vx_global * np.sin(yaw) + vy_global * np.cos(yaw)
        x[2] = wz
        x[3] = states[0]
        x[4] = states[1]
        x[5] = states[2]
        # print("yaw: ", yaw, "e_yaw: ", states[0])
        
        self.x = x

        return 

    # def run_mpc(self):
    #     if self.mode != 1 or self.pose_msg is None:
    #         return
    #     if self.first_mpc_run:
    #         lamb = 1e-7
    #         A, B, Error = Regression(self.xpid, self.upid, lamb)
    #         n = 6
    #         d = 2
    #         N = 14
    #         vt = 0.8
    #         mpcParam, ltvmpcParam = initMPCParams(n, d, N, vt)
    #         mpcParam.A = A
    #         mpcParam.B = B
    #         self.mpc = MPC(mpcParam)
    #         self.first_mpc_run = False
    #     self.mpc.solve(self.x)
    #     self.apply_control(self.mpc.get_control()[0], self.mpc.get_control()[1], self.vt)

    def run_pid(self):
        if self.pose_msg is None or self.mode != 0:
            return
        if self.x[4] > 1:
            self.flag = True
        if self.flag and self.x[4] < 0.1:
            self.flag = False
            self.get_logger().info(f"Lap {self.lap} Finished")
            self.lap += 1

            if self.lap >= self.pid_laps:
                self.mode = 1
                self.get_logger().info("Switching to MPC")
                np.savetxt("map/pid_x.csv", self.xpid, delimiter=",", fmt='%.4f')
                np.savetxt("map/pid_u.csv", self.upid, delimiter=",", fmt='%.4f')
                self.destroy_node()

            # ss = np.array(self.ss)
            # print(ss.shape, ss)
            # np.savetxt("map/pid_ss.csv", ss, delimiter=",", fmt='%.4f')
        #==== Solve PID
        self.pid_solve(self.x)
        #==== Apply Control
        self.apply_control(self.uPred[0, 0], self.uPred[0, 1], self.vt)

    def pid_init(self, vt):
        self.vt = vt
        self.uPred = np.zeros([1,2])

    def pid_solve(self, x0):
        p_acc = 1.5
        p_longitudinal_dist = 2.5
        p_yaw = -1.5
        self.uPred[0, 0] = p_longitudinal_dist * x0[5] + p_yaw * x0[3]
        self.uPred[0, 1] = p_acc * (self.vt - x0[0])

    def apply_control(self, steer, accel, vx):
        vel = vx + accel * self.dt
        
        steer = np.clip(steer, -self.car.STEER_MAX, self.car.STEER_MAX)
        self.get_logger().info(f"accel_cmd: {accel}, vel_cmd: {vel}, steer_cmd: {steer}")
        drive_msg = AckermannDriveStamped()
        drive_msg.header.stamp = self.get_clock().now().to_msg()
        drive_msg.header.frame_id = 'base_link'
        drive_msg.drive.steering_angle = steer
        drive_msg.drive.steering_angle_velocity = 1.0
        # if abs(steer) > 0.35:
        #     vel = 0.5
        drive_msg.drive.speed = vel
        # drive_msg.drive.acceleration = accel
        self.drive_pub.publish(drive_msg)
        
        if self.xpid is None:
            self.xpid = self.x.reshape(1, -1)
            self.upid = self.uPred[0].reshape(1, -1)
        else:
            self.xpid = np.concatenate([self.xpid, self.x.reshape(1, -1)], axis=0)
            
            self.upid = np.concatenate([self.upid, self.uPred[0].reshape(1, -1)], axis=0)
            print(self.xpid.shape, self.upid.shape)
        # print(self.x, self.last_x)


    def publish_waypoints(self, waypoints):
        # if len(self.waypoints) == 0:
        #     return
        
        markerArray = MarkerArray()
        for i, wp in enumerate(waypoints):
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.id = i
            marker.type = marker.SPHERE
            marker.action = marker.ADD
            marker.pose.position.x = float(wp[0])
            marker.pose.position.y = float(wp[1])
            marker.pose.position.z = 0.0
            marker.pose.orientation.x = 0.0
            marker.pose.orientation.y = 0.0
            marker.pose.orientation.z = 0.0
            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.1
            marker.scale.y = 0.1
            marker.scale.z = 0.1
            marker.color.a = 1.0
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.0
            markerArray.markers.append(marker)
        self.waypoints_publisher.publish(markerArray)


def main(args=None):
    rclpy.init(args=args)
    print("MPC Initialized")
    mpc_node = MPC()
    rclpy.spin(mpc_node)

    mpc_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
