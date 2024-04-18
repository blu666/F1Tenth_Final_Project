#!/usr/bin/env python3
import numpy as np
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
import scipy
from scipy.spatial.transform import Rotation as R
import tf2_ros
from visualization_msgs.msg import Marker, MarkerArray
from rclpy.parameter import Parameter, ParameterType
from sensor_msgs.msg import PointCloud
from car_params import CarParams
from track import Track ## TODO: implement Track class
# from rrt_star import RRT_Star


@dataclass
class SS_Sample:
    x: np.ndarray   # [x, y, yaw, v, yaw_dot, slip_angle]
    u: np.ndarray   # [accel, steer]
    s: float        # theta, progress along track
    time: int       # time step
    iter: int       # iteration
    cost: int

# class def for RRT
class LMPC(Node):
    def __init__(self):
        super().__init__('lmpc_node')
        # Create pub sub
        self.create_subscription(Odometry, 'ego_racecar/odom', self.odom_callback, 10)
        self.drive_publisher = self.create_publisher(AckermannDriveStamped, '/drive', 10)

        self.map_to_car_rotation = None
        self.map_to_car_translation = None
        
        self.first_run = True
        self.SS = None
        self.use_dynamics = True

        # global variables
        self.s_prev = 0
        self.s_curr = 0
        self.time = 0
        self.iter = 0

        self.curr_traj = []
        self.QPSol = None
        self.terminal_state_pred = None
        
        # Params: TODO: set_params
        self.car = CarParams()
        self.nx = 6 # dim of state space [x, y, yaw, v, omega, slip]
        self.nu = 2 # dim of control space
        self.ts = 0.01 # time step
        self.N = 16 # prediction horizon
        self.VEL_THRESH = 0.8
        self.K_NEAR = 16

    def lmpc_run(self):
        # line 387: run
        """
        Call lmpc_run() in odom_callback
        """
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

        self.terminal_state_pred = ...
        self.s_prev = self.s_curr
        self.time += 1
        self.first_run = False

    def odom_callback(self, pose_msg: Odometry):
        current_pose = np.array([pose_msg.pose.pose.position.x, pose_msg.pose.pose.position.y])
        current_heading = R.from_quat([pose_msg.pose.pose.orientation.x, pose_msg.pose.pose.orientation.y, pose_msg.pose.pose.orientation.z, pose_msg.pose.pose.orientation.w])
        self.map_to_car_translation = np.array([pose_msg.pose.pose.position.x, pose_msg.pose.pose.position.y, pose_msg.pose.pose.position.z])
        self.map_to_car_rotation = R.from_quat([pose_msg.pose.pose.orientation.x, pose_msg.pose.pose.orientation.y, pose_msg.pose.pose.orientation.z, pose_msg.pose.pose.orientation.w])

        s_curr = Track.findTheta(current_pose[0], current_pose[1], 0, True)
        yaw = ...
        vel = ...
        yawdot = ...
        slip_angle = ...

        if (not self.use_dynamics) and (vel > self.VEL_THRESH):
            self.use_dynamics = True
        elif(self.use_dynamics) and (vel < self.VEL_THRESH):
            self.use_dynamics = False
        # if (vel > 4.5):

        self.lmpc_run()

    def init_SS(self, data_file: str):
        # line 244: init_SS_from_data
        # Read data from csv file
        with open(data_file, 'r') as f:
            data = np.loadtxt(f, delimiter=',') # (time_steps, 8)
            # time, x, y, yaw, v, acc_cmd, steer_cmd, s_curr
        
        for i in range(data.shape[0]):
            pass
        # check if starting new lab
        # TODO: Design storing structure for SS
        pass

    def select_terminal_candidate(self):
        # line 456: select_terminal_candidate
        if self.first_run:
            return self.SS[-1][self.N].x
        else:
            return self.terminal_state_pred

    def add_point(self):
        # line 465: add_point
        point = SS_Sample()

        point.x = ...
        point.u = ...
        point.s = self.s_curr
        point.iter = self.iter
        point.time = self.time


    def select_convex_ss(self, iter_start, iter_end, s):
        # line 477: select_convex_safe_set
        convex_ss = []
        for it in range(iter_start, iter_end):
            nearest_idx = self.find_nearest_point(self.SS[it], s)
            lap_cost = self.SS[it][0].cost

            if self.K_NEAR % 2:
                start_idx = nearest_idx - (self.K_NEAR-1) // 2
                end_idx = nearest_idx + (self.K_NEAR-1) // 2
            else:
                start_idx = nearest_idx - self.K_NEAR // 2 + 1
                end_idx = nearest_idx + self.K_NEAR // 2

            curr_set = []
            if start_idx < 0:
                for i in range(start_idx + len(self.SS[it]), len(self.SS[it])):
                    curr_set.append(self.SS[it][i])
                    curr_set[-1].cost += lap_cost
                for i in range(0, end_idx):
                    curr_set.append(self.SS[it][i])
                if len(curr_set) != self.K_NEAR:
                    print("Error: curr_set length not equal to K_NEAR")
            elif end_idx+1 >= len(self.SS[it]):
                for i in range(start_idx, len(self.SS[it])):
                    curr_set.append(self.SS[it][i])
                    curr_set[-1].cost += lap_cost
                for i in range(0, end_idx - len(self.SS[it])):
                    curr_set.append(self.SS[it][i])
                if len(curr_set) != self.K_NEAR:
                    print("Error: curr_set length not equal to K_NEAR")
            else:
                for i in range(start_idx, end_idx+1):
                    curr_set.append(self.SS[it][i])
            convex_ss.extend(curr_set)
        return convex_ss


    def find_nearest_point(self, trajectory, s):
        # line 524: find_nearest_point
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
        # line 536: update_cost_to_go
        trajectory[-1].cost = 0
        for i in range(len(trajectory)-2, -1, -1):
            trajectory[i].cost = trajectory[i+1].cost + 1

    def track_to_global(self, e_y, e_yaw, s):
        # line 557: track_to_global
        dx_ds = Track.x_eval_d(s)
        dy_ds = Track.y_eval_d(s)

        proj = np.array([Track.x_eval(s), Track.y_eval(s)])
        temp = np.array([-dy_ds, dx_ds])
        temp = temp / np.linalg.norm(temp)
        pos = proj + temp * e_y
        yaw = e_yaw + np.arctan2(dy_ds, dx_ds)
        return np.array([pos[0], pos[1], yaw])

    def global_to_track(self, x, y, yaw, s):
        # line 543: global_to_track
        x_proj = Track.x_eval(s)
        y_proj = Track.y_eval(s)
        e_y = np.sqrt((x - x_proj)**2 + (y - y_proj)**2)
        dx_ds = Track.x_eval_d(s)
        dy_ds = Track.y_eval_d(s)
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
        yaw = x_op[2]
        v = x_op[3]
        accel = u_op[0]
        steer = u_op[1]
        yaw_dot = x_op[4]
        slip_angle = x_op[5]
        
        dynamics = np.zeros(6)
        
        A = np.zeros((self.nx, self.nx))
        B = np.zeros((self.nx, self.nu))
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

        h = np.zeros((6, 1))
        aux = np.zeros((self.nx+self.nx, self.nx+self.nx), dtype=float)
        M = np.zeros((self.nx+self.nx, self.nx+self.nx), dtype=float)
        M12 = np.zeros((self.nx, self.nx), dtype=float)
        
        aux[:self.nx, :self.nx] = A
        aux[:self.nx, self.nx:] = B
        M = scipy.linalg.expm(aux * self.ts) # TODO: LINE 680 HOW TO COMPUTE MATRIX EXPONENTIAL
        M12 = M[:self.nx, self.nx:self.nx+self.nx]
        h = dynamics.reshape(6, 1) - (A @ x_op + B @ u_op)
        
        Ad = scipy.linalg.expm(A * self.ts)
        Bd = M12 @ B
        hd = M12 @ h
        return Ad, Bd, hd

    
        
    def wrap_angle(self, angle, ref_angle):
        # line 688: wrap_angle
        while angle - ref_angle > np.pi:
            angle -= 2*np.pi
        while angle - ref_angle < -np.pi:
            angle += 2*np.pi
        return angle

    def solve_MPC(self, terminal_candidate):
        # line 693: solve_MPC
        s_t = ...

    def apply_control(self):
        # line 938: apply_control
        accel = ...
        steer = ...

        self.get_logger().info(f"accel_cmd: {accel}, steer_cmd: {steer}, slip_angle: {self.slip_angle}")
        steer = np.clip(steer, -self.car.steer_max, self.car.steer_max)

        drive_msg = AckermannDriveStamped()
        drive_msg.header.stamp = self.get_clock().now().to_msg()
        drive_msg.header.frame_id = 'base_link'
        drive_msg.drive.steering_angle = steer
        drive_msg.drive.steering_angle_velocity = 1.0
        drive_msg.drive.acceleration = accel
        self.drive_publisher.publish(drive_msg)
    

def main(args=None):
    rclpy.init(args=args)
    print("RRT Initialized")
    lmpc_node = LMPC()
    rclpy.spin(lmpc_node)

    lmpc_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()