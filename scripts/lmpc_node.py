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
from utils.utilities import Regression, load_init_ss
from utils.initControllerParameters import initLMPCParams
from utils.PredictiveModel import PredictiveModel
from scipy import sparse
import cvxpy as cp
from osqp import OSQP
from utils.PredictiveControllers import LMPC
from rclpy.time import Duration

# class def for RRT
class ControllerNode(Node):
    def __init__(self):
        super().__init__('lmpc_node')
        #==== Load Track (in first odom callback)
        self.Track = None

        #==== LMPC Params
        self.N = 20                                    # Horizon length
        self.n = 6; self.d = 2                            # State and Input dimension
        self.dt = 1. / 20.
        vt = 0.8
        self.car: CarParams = load_default_car_params()
        
        #==== LMPC Components
        self.lmpcpredictiveModel = None
        self.lmpc = None
        
        #==== Initialize Variables
        self.x_cl = []              # States for a closed loop. [vx, vy, wz, epsi, s, ey]
        self.x_cl_glob = []         # States for a closed loop in global frame. [vx, vy, wz, psi, X, Y]
        self.u_cl = []              # Control inputs for a closed loop. [delta, a]
        self.first_run = True       # if first run, initialize x_cl and x_cl_glob with initial odom callback
        self.time = 0               # record lapping time steps
        self.starting_timestamp = None 
        self.lap = 0                # record laps
        self.s_prev = 0             # record previous s
        self.odom: Odometry = None  # Odometry message

        #==== Create pub sub
        self.create_subscription(Odometry, '/ego_racecar/odom', self.odom_callback, 10)
        self.drive_publisher = self.create_publisher(AckermannDriveStamped, '/drive', 10)
        self.waypoints_publisher = self.create_publisher(MarkerArray, '/pure_pursuit/waypoints', 10)
        self.testpoint_publisher = self.create_publisher(MarkerArray, '/pure_pursuit/testpoints', 10)
        self.selected_publisher = self.create_publisher(MarkerArray, '/pure_pursuit/selected', 10)
        self.create_timer(self.dt, self.lmpc_run) # RUN LMPC WITH FIXED RATE
        self.get_logger().info("@=>Init: LMPC Node Initialized")
        
        

    def lmpc_run(self):
        if self.first_run or self.odom is None:
            #===> Not yet initialized
            return
        if self.starting_timestamp is None:
            self.starting_timestamp = self.get_clock().now()
            # end_time = self.get_clock().now()
            # print("[!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1]", (self.starting_timestamp - end_time).nanoseconds * 10e-9)
        curr_odom = deepcopy(self.odom)
        X, Y = self.odom.pose.pose.position.x, self.odom.pose.pose.position.y
        self.map_to_car_translation = np.array([self.odom.pose.pose.position.x,
                                                self.odom.pose.pose.position.y,
                                                self.odom.pose.pose.position.z])
        self.map_to_car_rotation = R.from_quat([self.odom.pose.pose.orientation.x,
                                                self.odom.pose.pose.orientation.y,
                                                self.odom.pose.pose.orientation.z,
                                                self.odom.pose.pose.orientation.w])

        yaw = self.map_to_car_rotation.as_euler('zyx')[0]
        # dpos = np.array([np.cos(yaw), np.sin(yaw)])
        # dpos = normalize_vector(dpos)
        # X += dpos[0] * 0.2
        # Y += dpos[1] * 0.2
        vx, vy = self.odom.twist.twist.linear.x, self.odom.twist.twist.linear.y# + np.random.randn() * 1e-6
        wz = self.odom.twist.twist.angular.z
        epsi, s_curr, ey, _ = self.Track.get_states(X, Y, yaw)
        
        """
        TODO 要不要在第二圈继续累加 还是从0开始
        
        """
        # if (self.lap + 1) % 2 == 0:
        #     s_state = s_curr + self.Track.length # NEW: ON EVEN LAP, ACCUMULATE S
        # else:
        #     s_state = s_curr
        s_state = s_curr
        self.xt = np.array([vx, vy, wz, epsi, s_state, ey])
        self.xt_glob = np.array([vx, vy, wz, yaw, X, Y])
        
        
        self.lmpc.solve(self.xt)
        
        #==== Visualize ss point
        ss_points = self.lmpc.Succ_SS_PointSelectedTot
        pub_states = np.empty((ss_points.shape[1], 2))
        for i in range(ss_points.shape[1]):
            x, y, yaw = self.Track.track_to_global(ss_points[5, i], ss_points[3, i], ss_points[4, i])
            pub_states[i, 0] = x
            pub_states[i, 1] = y
        # print(pub_states.shape, ss_points.shape)
        self.publish_selected(pub_states)
        
        
        u = self.lmpc.get_control()
        self.lmpc.addPoint(self.xt, u) # at iteration j add data to SS^{j-1} 
        # self.get_logger().info("@=> states: xt {}".format(self.xt))
        #==== Check if the car has passed the starting line
        if s_curr - self.s_prev < -self.Track.length / 3.:
            end_time = self.get_clock().now()
            print("@=>>>>>>>>>>>>>>>>")
            print("@=>Lapping: Finished running lap {}, timesteps {}, time: {}".format(self.lap, self.time, -(self.starting_timestamp - end_time).nanoseconds * 10e-10))
            print("@=>>>>>>>>>>>>>>>>")
            self.starting_timestamp = end_time
            self.time = 0
            self.lap += 1
            
            if self.lap % 2 == 0:
                """
                NEW: IF EVEN LAP finished, add to save set.
                """
                #==== Lap finished, add to safe set
                self.lmpc.addTrajectory(np.array(self.x_cl), np.array(self.u_cl), np.array(self.x_cl_glob))
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
        else:
            self.time += 1
            #==== Record trajectory
            if self.lap % 2 == 1:
                self.xt[4] += self.Track.length
            self.x_cl.append(self.xt)
            self.x_cl_glob.append(self.xt_glob)
            self.u_cl.append(u.copy())
        self.s_prev = s_curr
        self.apply_control(u[0], u[1], vx)

        
        # self.plot_ss()

    def plot_ss(self):
        SS_list = self.lmpc.getCurrentSS() # list of (N, 6) numpy arrays
        ss_points = np.concatenate(SS_list, axis=0)
        # pub_states = ss_points[:, 4:6]
        pub_states = np.empty((ss_points.shape[0], 2))
        for i in range(ss_points.shape[0]):
            x, y, yaw = self.Track.track_to_global(ss_points[i, 5], ss_points[i, 3], ss_points[i, 4])
            pub_states[i, 0] = x
            pub_states[i, 1] = y
        self.publish_testpoints(pub_states)
        # # print(SS_combined.shape)
        # return SS_combined.shape
    
    def initialize_lmpc(self, N, n, d, track):
        x0_cls, u0_cls, x0_cl_globs = load_init_ss('./map/initial_ss.csv', 5, self.Track.length)
        # ss_points = x0_cls[0]
        # pub_states = np.empty((ss_points.shape[0], 2))
        # for i in range(ss_points.shape[0]):
        #     x, y, yaw = self.Track.track_to_global(ss_points[i, 5], ss_points[i, 3], ss_points[i, 4])
        #     pub_states[i, 0] = x
        #     pub_states[i, 1] = y
        # print(pub_states.shape, ss_points.shape)
        # self.publish_testpoints(pub_states)
        # self.publish_testpoints(x0_cl_globs[0][:, 4:6])
        # self.get_logger().info("@=>Init: initial safety set lenghth: {}".format(len(x0_cl)))
        # mpcParam, ltvmpcParam = initMPCParams(n, d, N, vt)
        numSS_it, numSS_Points, _, _, QterminalSlack, lmpcParameters = initLMPCParams(track, N) # TODO: change from map to self.Track
        self.lmpcpredictiveModel = PredictiveModel(n, d, track, 4)
        for i in range(0, 4): # add trajectories used for model learning
            self.lmpcpredictiveModel.addTrajectory(x0_cls[i],u0_cls[i])
        x0_cls_all = np.concatenate(x0_cls, axis=0)
        u0_cls_all = np.concatenate(u0_cls, axis=0)
        A, B, Error = Regression(x0_cls_all, u0_cls_all, lamb=1e-6)
        print("error", Error)
        lmpcParameters.A = A
        lmpcParameters.B = B
        lmpcParameters.timeVarying     = False
        self.lmpc = LMPC(numSS_Points, numSS_it, QterminalSlack, lmpcParameters, self.lmpcpredictiveModel)
        for i in range(0, 4): # add trajectories for safe set
            self.lmpc.addTrajectory(x0_cls[i], u0_cls[i], x0_cl_globs[i])
        return

    def odom_callback(self, pose_msg: Odometry):
        #==== For first run, initilize Track and LMPC.
        self.odom = pose_msg
        if self.first_run:
            #==== Initialize Track & reset starting point to spawn point
            self.Track = Track("./map/race3/centerline.csv", None)
            self.Track.reset_starting_point(pose_msg.pose.pose.position.x,
                                            pose_msg.pose.pose.position.y,
                                            refine=True)
            self.get_logger().info("@=>Init: Track Loaded")
            #==== Initialize LMPC
            self.initialize_lmpc(self.N, self.n, self.d, self.Track)
            self.get_logger().info("@=>Init: LMPC Initialized")
            self.first_run = False
            return 
        
        #==== Update car state and call LMPC
        

    def apply_control(self, steer, accel, vx):
        # line 938: apply_control
        vel = vx + accel * self.dt
        
        steer = np.clip(steer, -self.car.STEER_MAX, self.car.STEER_MAX)
        # self.get_logger().info("accel_cmd: {:.6f}, vel_cmd: {:.6f}, steer_cmd: {:.6f}".format(accel, vel, steer))
        drive_msg = AckermannDriveStamped()
        drive_msg.header.stamp = self.get_clock().now().to_msg()
        drive_msg.header.frame_id = 'base_link'
        drive_msg.drive.steering_angle = steer
        # drive_msg.drive.steering_angle_velocity = 1.0
        drive_msg.drive.speed = vel
        # drive_msg.drive.acceleration = accel
        self.drive_publisher.publish(drive_msg)
    
    def publish_testpoints(self, testpoints):
        markerArray = MarkerArray()
        for i, tp in enumerate(testpoints):
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.id = i
            marker.type = marker.SPHERE
            marker.action = marker.ADD
            marker.pose.position.x = tp[0]
            marker.pose.position.y = tp[1]
            marker.pose.position.z = 0.0
            marker.pose.orientation.x = 0.0
            marker.pose.orientation.y = 0.0
            marker.pose.orientation.z = 0.0
            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.1
            marker.scale.y = 0.1
            marker.scale.z = 0.1
            marker.color.a = 0.2
            marker.color.r = 0.0
            marker.color.g = 0.0
            marker.color.b = 1.0
            markerArray.markers.append(marker)
        self.testpoint_publisher.publish(markerArray)
    
    def publish_selected(self, testpoints):
        markerArray = MarkerArray()
        for i, tp in enumerate(testpoints):
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.id = i
            marker.type = marker.SPHERE
            marker.action = marker.ADD
            marker.pose.position.x = tp[0]
            marker.pose.position.y = tp[1]
            marker.pose.position.z = 0.0
            marker.pose.orientation.x = 0.0
            marker.pose.orientation.y = 0.0
            marker.pose.orientation.z = 0.0
            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.21
            marker.scale.y = 0.21
            marker.scale.z = 0.21
            marker.color.a = 1.0
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.0
            markerArray.markers.append(marker)
        self.selected_publisher.publish(markerArray)


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