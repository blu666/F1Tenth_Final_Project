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
from utils.initControllerParameters import initMPCParams, initLMPCParams
from utils.PredictiveModel import PredictiveModel
from scipy import sparse
import cvxpy as cp
from osqp import OSQP
from utils.PredictiveControllers import LMPC


# class def for RRT
class ControllerNode(Node):
    def __init__(self):
        super().__init__('lmpc_node')
        #==== Load Track
        self.Track = Track("map/refined_centerline.csv", initialized=True)

        #==== LMPC Params
        N = 14                                    # Horizon length
        n = 6;   d = 2                            # State and Input dimension
        x0 = np.array([0.5, 0, 0, 0, 0, 0])       # Initial condition
        self.xS = [x0, x0]
        self.dt = 1. / 30.
        vt = 0.8
        self.car: CarParams = load_default_car_params()
        
        #==== Initialize LMPC
        xPID_cl, uPID_cl, xPID_cl_glob = load_init_ss('./map/initial_ss.csv')
        # mpcParam, ltvmpcParam = initMPCParams(n, d, N, vt)
        numSS_it, numSS_Points, _, _, QterminalSlack, lmpcParameters = initLMPCParams(self.Track, N) # TODO: change from map to self.Track
        self.lmpcpredictiveModel = PredictiveModel(n, d, self.Track, 4)
        for i in range(0,4): # add trajectories used for model learning
            self.lmpcpredictiveModel.addTrajectory(xPID_cl,uPID_cl)
        lmpcParameters.timeVarying     = True 
        self.lmpc = LMPC(numSS_Points, numSS_it, QterminalSlack, lmpcParameters, self.lmpcpredictiveModel)
        for i in range(0,4): # add trajectories for safe set
            self.lmpc.addTrajectory( xPID_cl, uPID_cl, xPID_cl_glob)
        print("Initialized LMPC")
        
        #==== Initialize Variables
        self.x_cl = []          # States for a closed loop. [vx, vy, wz, epsi, s, ey]
        self.x_cl_glob = []     # States for a closed loop in global frame. [vx, vy, wz, psi, X, Y]
        self.u_cl = []          # Control inputs for a closed loop. [delta, a]
        self.first_run = True   # if first run, initialize x_cl and x_cl_glob with initial odom callback
        self.time = 0           # record lapping time steps
        self.lap = 0            # record laps
        self.s_prev = 0         # record previous s        

        #==== Create pub sub
        self.create_subscription(Odometry, 'ego_racecar/odom', self.odom_callback, 10)
        self.drive_publisher = self.create_publisher(AckermannDriveStamped, '/drive', 10)
        self.waypoints_publisher = self.create_publisher(MarkerArray, '/pure_pursuit/waypoints', 10)
        self.create_timer(1 / 20.0, self.lmpc_run)
        self.get_logger().info("LMPC Node Initialized")


    def lmpc_run(self):
        pass
    

    def odom_callback(self, pose_msg: Odometry):
        #==== For first run, initilize x_cl and x_cl_glob
        # if self.first_run:
        #     self.x_cl = np.array([0, 0, ])
        #     self.x_cl_glob = np.array([pose_msg.pose.pose.position.x, pose_msg.pose.position.y, 0, 0, 0, 0])
        #     self.first_run = False
        #     return 
        #==== Update car state and call LMPC
        X, Y = pose_msg.pose.pose.position.x, pose_msg.pose.pose.position.y
        self.map_to_car_translation = np.array([pose_msg.pose.pose.position.x,
                                                pose_msg.pose.pose.position.y,
                                                pose_msg.pose.pose.position.z])
        self.map_to_car_rotation = R.from_quat([pose_msg.pose.pose.orientation.x,
                                                pose_msg.pose.pose.orientation.y,
                                                pose_msg.pose.pose.orientation.z,
                                                pose_msg.pose.pose.orientation.w])
        yaw = self.map_to_car_rotation.as_euler('zyx')[0]
        vx, vy = pose_msg.twist.twist.linear.x, pose_msg.twist.twist.linear.y
        wz = pose_msg.twist.twist.angular.z
        epsi, s_curr, ey, closepoint = self.Track.get_states(X, Y, yaw)
        self.xt = np.array([vx, vy, wz, epsi, s_curr, ey])
        self.xt_glob = np.array([vx, vy, wz, yaw, X, Y])
        self.lmpc.solve(self.xt)
        u = self.lmpc.get_control()
        self.lmpc.addPoint(self.x_t, u) # at iteration j add data to SS^{j-1} 
        #==== Check if the car has passed the starting line
        if s_curr - self.s_prev < -self.Track.length / 3.:
            print("@=> Finished runing lap {}, time {}".format(self.lap, self.time))
            self.time = 0
            self.lap += 1
            #==== Lap finished, add to safe set
            self.lmpc.addTrajectory(self.x_cl, self.u_cl, self.x_cl_glob)
            #==== reset recorded trajectory 
            self.x_cl = [self.xt]
            self.x_cl_glob = [self.xt_glob]
            self.u_cl = [u.copy()]
        else:
            self.time += 1
            #==== Record trajectory
            self.x_cl.append(self.xt)
            self.x_cl_glob.append(self.xt_glob)
            self.u_cl.append(u.copy())
        self.s_prev = s_curr
        self.apply_control(u[0], u[1], vx)

    def apply_control(self, steer, accel, vx):
        # line 938: apply_control
        vel = vx + accel * self.dt
        self.get_logger().info(f"accel_cmd: {accel}, vel_cmd: {vel}, steer_cmd: {steer}")
        steer = np.clip(steer, -self.car.STEER_MAX, self.car.STEER_MAX)

        drive_msg = AckermannDriveStamped()
        drive_msg.header.stamp = self.get_clock().now().to_msg()
        drive_msg.header.frame_id = 'base_link'
        drive_msg.drive.steering_angle = steer
        # drive_msg.drive.steering_angle_velocity = 1.0
        drive_msg.drive.speed = vel
        # drive_msg.drive.acceleration = accel
        self.drive_publisher.publish(drive_msg)

def normalize_vector(vec):
    norm = np.linalg.norm(vec)
    return vec / norm

def main(args=None):
    rclpy.init(args=args)
    # print("RRT Initialized")
    lmpc_node = ControllerNode()
    rclpy.spin(lmpc_node)

    lmpc_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()