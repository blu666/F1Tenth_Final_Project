#!/usr/bin/env python3
import numpy as np
import math

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Point
from ackermann_msgs.msg import AckermannDriveStamped, AckermannDrive
from nav_msgs.msg import OccupancyGrid, Odometry


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
from osqp import OSQP


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
        
        
        self.map_to_car_rotation = None
        self.map_to_car_translation = None
        
        self.first_run = True
        self.SS = None
        self.use_dynamics = True

        self.Track = Track("map/reassigned_centerline.csv")

        # global variables
        self.s_prev = 0
        self.s_curr = 0
        self.time = 0
        self.iter = 0
        self.car_pos = np.zeros(2)
        self.yaw = 0
        self.vel = 0
        self.yawdot = 0
        self.slip_angle = 0

        self.curr_traj = []
        self.QPSol = None
        self.terminal_state_pred = None
        
        # Params: TODO: set_params
        self.car = load_default_car_params()
        self.nx = 6 # dim of state space [x, y, yaw, v, omega, slip]
        self.nu = 2 # dim of control space
        self.ts = 0.01 # time steps

        # Load SS from data
        self.init_SS("map/levine/inital_ss.csv")
        
        # setup osqp
        self.osqp = OSQP()

        HessianMatrix = sparse.csr_matrix(((self.car.N+1)*self.nx + self.car.N*self.nu + self.car.N+1 + 2*self.car.K_NEAR + self.nx, 
                                          (self.car.N+1)*self.nx + self.car.N*self.nu + self.car.N+1 + 2*self.car.K_NEAR + self.nx))
        
        constraintMatrix = sparse.csr_matrix(((self.car.N+1)*self.nx + 2*(self.car.N+1) + self.car.N*self.nu + self.car.N+1 + self.car.N+1 + 2*self.car.K_NEAR + 2*self.nx + 1,
                                             (self.car.N+1)*self.nx + self.car.N*self.nu + self.car.N+1 + 2*self.car.K_NEAR + self.nx))
        
        gradient = np.zeros((self.car.N+1)*self.nx + self.car.N*self.nu + self.car.N+1 + 2*self.car.K_NEAR + self.nx)

        lower = np.zeros((self.car.N+1)*self.nx + 2*(self.car.N+1) + self.car.N*self.nu + self.car.N+1 + self.car.N+1 + 2*self.car.K_NEAR + 2*self.nx + 1)
        upper = np.zeros((self.car.N+1)*self.nx + 2*(self.car.N+1) + self.car.N*self.nu + self.car.N+1 + self.car.N+1 + 2*self.car.K_NEAR + 2*self.nx + 1)
        self.osqp.setup(P=HessianMatrix, q=gradient, A=constraintMatrix, l=lower, u=upper, warm_start=True)

        # Create pub sub
        self.create_subscription(Odometry, 'ego_racecar/odom', self.odom_callback, 10)
        self.drive_publisher = self.create_publisher(AckermannDriveStamped, '/drive', 10)
        self.create_timer(1 / 20.0, self.lmpc_run)
        self.get_logger().info("LMPC Node Initialized")


    def lmpc_run(self):
        # line 387: run
        """
        Call lmpc_run() in odom_callback
        """
        if self.first_run:
            # reset QP solution
            self.QPSol = np.zeros((self.car.N+1)*self.nx + self.car.N*self.nu + self.nx*(self.car.N+1) + 2*(self.car.K_NEAR+1))
            for i in range(self.car.N+1):
                self.QPSol[i*self.nx:i*self.nx+self.nx] = self.SS[self.iter][i].x
            for i in range(self.car.N):
                self.QPSol[(self.car.N+1)*self.nx+i*self.nu:(self.car.N+1)*self.nx+i*self.nu+self.nu] = self.SS[self.iter][i].u

        # check if new lap
        if self.s_curr - self.s_prev < -self.Track.length/2:
            self.iter += 1
            self.update_cost_to_go(self.SS[self.iter])
            self.SS.append(self.curr_traj.copy())
            self.curr_traj = []
            self.time = 0

        # select terminal candidate
        terminal_candidate = self.select_terminal_candidate()
        self.solve_MPC(terminal_candidate)
        self.apply_control()
        self.add_point()

        self.terminal_state_pred = self.QPSol[self.car.N*self.nx:(self.car.N+1)*self.nx]
        self.s_prev = self.s_curr
        self.time += 1
        self.first_run = False
        self.get_logger().info("LMPC Run iter: {}, time: {0}".format(self.iter, self.time))

    def odom_callback(self, pose_msg: Odometry):
        current_pose = np.array([pose_msg.pose.pose.position.x, pose_msg.pose.pose.position.y])
        current_heading = R.from_quat([pose_msg.pose.pose.orientation.x, pose_msg.pose.pose.orientation.y, pose_msg.pose.pose.orientation.z, pose_msg.pose.pose.orientation.w])
        self.map_to_car_translation = np.array([pose_msg.pose.pose.position.x, pose_msg.pose.pose.position.y, pose_msg.pose.pose.position.z])
        self.map_to_car_rotation = R.from_quat([pose_msg.pose.pose.orientation.x, pose_msg.pose.pose.orientation.y, pose_msg.pose.pose.orientation.z, pose_msg.pose.pose.orientation.w])

        s_curr = self.Track.find_theta(current_pose)
        self.yaw = current_heading.as_euler('zyx')[0]
        self.vel = np.linalg.norm([pose_msg.twist.twist.linear.x, pose_msg.twist.twist.linear.y])
        self.yawdot = pose_msg.twist.twist.angular.z
        self.slip_angle = np.arctan2(pose_msg.twist.twist.linear.y, pose_msg.twist.twist.linear.x)

        if (not self.use_dynamics) and (self.vel > self.car.DYNA_VEL_THRESH):
            self.use_dynamics = True
        elif(self.use_dynamics) and (self.vel < self.car.DYNA_VEL_THRESH):
            self.use_dynamics = False
        # if (vel > 4.5):

        # self.lmpc_run()

    def init_SS(self, data_file: str):
        # line 244: init_SS_from_data
        # Read data from csv file
        self.SS = []
        # header = "time, x, y, yaw, vel, acc_cmd, steer_cmd, s, lap"
        data: np.ndarray = np.loadtxt(data_file, delimiter=',', skiprows=1) # (time_steps, 8)
        traj = []
        prev_time = -1
        iteration = 0
        for i in range(data.shape[0]):
            sample = SS_Sample(time=int(data[i, 0]), 
                               x=np.array([data[i, 1], data[i, 2], data[i, 3], data[i, 4], 0, 0]), 
                               u=np.array([data[i, 5], data[i, 6]]), 
                               s=data[i, 7], 
                               iter=iteration,
                               cost=0)
            if sample.time < prev_time:
                iteration += 1
                traj = self.update_cost_to_go(traj)
                self.SS.append(deepcopy(traj))
                traj = []
            sample.iter = iteration
            traj.append(sample)
            prev_time = sample.time
        traj = self.update_cost_to_go(traj)
        self.SS.append(deepcopy(traj))

    def select_terminal_candidate(self):
        # line 456: select_terminal_candidate
        if self.first_run:
            return self.SS[-1][self.car.N].x
        else:
            return self.terminal_state_pred

    def add_point(self):
        # line 465: add_point
        point = SS_Sample()

        point.x = np.array([self.car_pos[0], self.car_pos[1], self.yaw, self.vel, self.yawdot, self.slip_angle])
        point.u = self.QPSol[self.nx*(self.car.N+1):self.nx*(self.car.N+1)+self.nu]
        point.s = self.s_curr
        point.iter = self.iter
        point.time = self.time


    def select_convex_ss(self, iter_start, iter_end, s):
        # line 477: select_convex_safe_set
        convex_ss = []
        for it in range(iter_start, iter_end):
            nearest_idx = self.find_nearest_point(self.SS[it], s)
            lap_cost = self.SS[it][0].cost

            if self.car.K_NEAR % 2:
                start_idx = nearest_idx - (self.car.K_NEAR-1) // 2
                end_idx = nearest_idx + (self.car.K_NEAR-1) // 2
            else:
                start_idx = nearest_idx - self.car.K_NEAR // 2 + 1
                end_idx = nearest_idx + self.car.K_NEAR // 2

            curr_set = []
            if start_idx < 0:
                for i in range(start_idx + len(self.SS[it]), len(self.SS[it])):
                    curr_set.append(self.SS[it][i])
                    curr_set[-1].cost += lap_cost
                for i in range(0, end_idx):
                    curr_set.append(self.SS[it][i])
                if len(curr_set) != self.car.K_NEAR:
                    print("Error: curr_set length not equal to K_NEAR")
            elif end_idx+1 >= len(self.SS[it]):
                for i in range(start_idx, len(self.SS[it])):
                    curr_set.append(self.SS[it][i])
                    curr_set[-1].cost += lap_cost
                for i in range(0, end_idx - len(self.SS[it])):
                    curr_set.append(self.SS[it][i])
                if len(curr_set) != self.car.K_NEAR:
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
        return trajectory

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
        s_t = self.Track.find_theta(terminal_candidate[:2])

        convex_ss = self.select_convex_ss(self.iter-2, self.iter-1, s_t)

        HessianMatrix = sparse.csr_matrix(((self.car.N+1)*self.nx + self.car.N*self.nu + self.car.N+1 + 2*self.car.K_NEAR + self.nx, 
                                          (self.car.N+1)*self.nx + self.car.N*self.nu + self.car.N+1 + 2*self.car.K_NEAR + self.nx))
        
        constraintMatrix = sparse.csr_matrix(((self.car.N+1)*self.nx + 2*(self.car.N+1) + self.car.N*self.nu + self.car.N+1 + self.car.N+1 + 2*self.car.K_NEAR + 2*self.nx + 1,
                                             (self.car.N+1)*self.nx + self.car.N*self.nu + self.car.N+1 + 2*self.car.K_NEAR + self.nx))
        
        gradient = np.zeros((self.car.N+1)*self.nx + self.car.N*self.nu + self.car.N+1 + 2*self.car.K_NEAR + self.nx)

        lower = np.zeros((self.car.N+1)*self.nx + 2*(self.car.N+1) + self.car.N*self.nu + self.car.N+1 + self.car.N+1 + 2*self.car.K_NEAR + 2*self.nx + 1)
        upper = np.zeros((self.car.N+1)*self.nx + 2*(self.car.N+1) + self.car.N*self.nu + self.car.N+1 + self.car.N+1 + 2*self.car.K_NEAR + 2*self.nx + 1)

        x_k_ref = np.zeros((self.nx, 1))
        u_k_ref = np.zeros((self.nu, 1))
        Ad = np.zeros((self.nx, self.nx))
        Bd = np.zeros((self.nx, self.nu))
        x0 = np.zeros((self.nx, 1))
        hd = np.zeros((self.nx, 1))

        if self.use_dynamics:
            x0 = np.array([self.car_pos[0], self.car_pos[1], self.yaw, self.vel, self.yaw_dot, self.slip_angle])
        else:
            x0 = np.array([self.car_pos[0], self.car_pos[1], self.yaw, self.vel, 0.0, 0.0])
        
        for i in range(len(convex_ss)):
            convex_ss[i].x[2] = self.wrap_angle(convex_ss[i].x[2], x0[2])
        
        for i in range(self.car.N):
            # line 726
            # wrap angle for previous QPSolution
            self.QPSol[i*self.nx+2] = self.wrap_angle(self.QPSol[i*self.nx+2], x0[2])

        for i in range(self.car.N+1):
            x_k_ref = self.QPSol[i*self.nx : i*self.nx+self.nx]
            u_k_ref = self.QPSol[(self.car.N+1)*self.nx+i*self.nu : (self.car.N+1)*self.nx+i*self.nu+self.nu]
            s_ref = self.Track.find_theta(x_k_ref[:2])

            Ad, Bd, hd = self.get_linearized_dynamics(Ad, Bd, hd, x_k_ref, u_k_ref, self.use_dynamics)

            # Hessian entries
            if (i > 0): # cost only depends on 1 to N, not on x0
                HessianMatrix.insert((self.car.N+1)*self.nx + self.car.N*self.nu + i, (self.car.N+1)*self.nx + self.car.N*self.nu + i, self.car.q_s)
            if (i < self.N):
                # for row in range(self.nu):
                HessianMatrix.insert((self.car.N+1)*self.nx + i*self.nu, (self.car.N+1)*self.nx + i*self.nu, self.car.r_accel)
                HessianMatrix.insert((self.car.N+1)*self.nx + i*self.nu + 1, (self.car.N+1)*self.nx + i*self.nu + 1, self.car.r_steer)
            
            if (i < self.N):
                # Ad
                for row in range(self.nx):
                    for col in range(self.nx):
                        constraintMatrix.insert((i+1)*self.nx + row, i*self.nx + col, Ad[row, col])
                # Bd
                for row in range(self.nx):
                    for col in range(self.nu):
                        constraintMatrix.insert((i+1)*self.nx + row, (self.car.N+1)*self.nx + i*self.nu + col, Bd[row, col])

                lower[(i+1)*self.nx : (i+1)*self.nx + self.nx] = -hd
                upper[(i+1)*self.nx : (i+1)*self.nx + self.nx] = -hd
            
            for row in range(self.nx):
                constraintMatrix.insert(i*self.nx + row, i*self.nx + row, -1.0)
            
            dx_dtheta = self.Track.x_eval_d(s_ref)
            dy_dtheta = self.Track.y_eval_d(s_ref)

            constraintMatrix.insert((self.car.N+1)*self.nx + 2*i, i*self.nx, -dy_dtheta)
            constraintMatrix.insert((self.car.N+1)*self.nx + 2*i, i*self.nx+1, dx_dtheta)
            constraintMatrix.insert((self.car.N+1)*self.nx + 2*i, (self.car.N+1)*self.nx + self.car.N*self.nu + i, 1.0)

            constraintMatrix.insert((self.car.N+1)*self.nx + 2*i+1, i*self.nx, -dy_dtheta)
            constraintMatrix.insert((self.car.N+1)*self.nx + 2*i+1, i*self.nx+1, dx_dtheta)
            constraintMatrix.insert((self.car.N+1)*self.nx + 2*i+1, (self.car.N+1)*self.nx + self.car.N*self.nu + i, -1.0)


            center_p = np.array([self.Track.x_eval(s_ref), self.Track.y_eval(s_ref)])
            left_tangent_p = center_p + self.Track.get_left_half_width(s_ref) * self.normalize_vector(np.array([-dy_dtheta, dx_dtheta]))
            right_tangent_p = center_p + self.Track.get_right_half_width(s_ref) * self.normalize_vector(np.array([dy_dtheta, -dx_dtheta]))

            C1 = -dy_dtheta * right_tangent_p[0] + dx_dtheta * right_tangent_p[1]
            C2 = -dy_dtheta * left_tangent_p[0] + dx_dtheta * left_tangent_p[1]

            lower[(self.car.N+1)*self.nx + 2*i] = min(C1, C2)
            upper[(self.car.N+1)*self.nx + 2*i] = np.inf

            lower[(self.car.N+1)*self.nx + 2*i+1] = -np.inf
            upper[(self.car.N+1)*self.nx + 2*i+1] = max(C1, C2)

            # u_min < u < u_max
            if (i<self.car.N):
                for row in range(self.nu):
                    constraintMatrix.insert((self.car.N+1)*self.nx + 2*(self.car.N+1) + i*self.nu + row, (self.car.N+1)*self.nx + i*self.nu + row, 1.0)
                lower[(self.car.N+1)*self.nx + 2*(self.car.N+1) + i*self.nu : (self.car.N+1)*self.nx + 2*(self.car.N+1) + i*self.nu + self.nu] = np.array([-self.car.DECELRATION_MAX, -self.car.STEER_MAX])
                upper[(self.car.N+1)*self.nx + 2*(self.car.N+1) + i*self.nu : (self.car.N+1)*self.nx + 2*(self.car.N+1) + i*self.nu + self.nu] = np.array([self.car.ACCELERATION_MAX, self.car.STEER_MAX])
            
            # max vel
            constraintMatrix.insert((self.car.N+1)*self.nx + 2*(self.car.N+1) + self.car.N*self.nu + i, i*self.nx+3, 1.0)
            lower[(self.car.N+1)*self.nx + 2*(self.car.N+1) + self.car.N*self.nu + i] = 0.0
            upper[(self.car.N+1)*self.nx + 2*(self.car.N+1) + self.car.N*self.nu + i] = self.car.VEL_MAX

            # s_k >= 0
            constraintMatrix.insert((self.car.N+1)*self.nx + 2*(self.car.N+1) + self.car.N*self.nu + self.car.N+1 + i, (self.car.N+1)*self.nx+self.car.N*self.nu+i, 1.0)
            lower[(self.car.N+1)*self.nx + 2*(self.car.N+1) + self.car.N*self.nu + self.car.N+1 + i] = 0.0
            upper[(self.car.N+1)*self.nx + 2*(self.car.N+1) + self.car.N*self.nu + self.car.N+1 + i] = np.inf

        num_constraint_sofar = (self.car.N+1)*self.nx + 2*(self.car.N+1) + self.car.N*self.nu + self.car.N+1 + self.car.N+1

        for i in range(2*self.car.K_NEAR):
            # lamda >= 0
            constraintMatrix.insert(num_constraint_sofar + i, (self.car.N+1)*self.nx + self.car.N*self.nu + i, 1.0)
            lower[num_constraint_sofar + i] = 0.0
            upper[num_constraint_sofar + i] = np.inf
        num_constraint_sofar += 2*self.car.K_NEAR

        # terminal state constraints
        # -s_t <= -x_{N+1} + linear_combination(lambda) <= s_t
        # 0 <= s_t - x_{N+1} + linear_combination(lambda) <= inf
        for i in range(2*self.K_NEAR):
            for state_idx in range(self.nx):
                constraintMatrix.insert(num_constraint_sofar + state_idx, (self.car.N+1)*self.nx + self.car.N*self.nu + self.car.N+1 + i, terminal_candidate[i].x[state_idx])
        for state_idx in range(self.nx):
            constraintMatrix.insert(num_constraint_sofar + state_idx, self.car.N*self.nx + state_idx, -1.0)
            constraintMatrix.insert(num_constraint_sofar + state_idx, (self.car.N+1)*self.nx + self.car.N*self.nu + 2*self.K_NEAR + state_idx, 1.0)

            lower[num_constraint_sofar + state_idx] = 0.0
            upper[num_constraint_sofar + state_idx] = np.inf

        num_constraint_sofar += self.nx

        # sum of lambda = 1
        for i in range(2*self.K_NEAR):
            constraintMatrix.insert(num_constraint_sofar, (self.car.N+1)*self.nx + self.car.N*self.nu + self.car.N+1 + i, 1.0)

        lower[num_constraint_sofar] = 1.0
        upper[num_constraint_sofar] = 1.0
        num_constraint_sofar += 1

        if num_constraint_sofar != (self.car.N+1)*self.nx + 2*(self.car.N+1) + self.car.N*self.nu + self.car.N+1 + self.car.N+1 + 2*self.K_NEAR + 2*self.nx + 1:
            print("Error: num_constraint_sofar not equal to expected value")

        lowest_cost1 = terminal_candidate[self.car.K_NEAR-1].cost
        lowest_cost2 = terminal_candidate[2*self.car.K_NEAR-1].cost

        for i in range(self.car.K_NEAR):
            gradient[(self.car.N+1)*self.nx + self.car.N*self.nu + self.car.N+1 + i] = terminal_candidate[i].cost - lowest_cost1
        for i in range(self.car.K_NEAR, 2*self.car.K_NEAR):
            gradient[(self.car.N+1)*self.nx + self.car.N*self.nu + self.car.N+1 + i] = terminal_candidate[i].cost - lowest_cost2

        for i in range(self.car.nx):
            HessianMatrix.insert((self.car.N+1)*self.nx + self.car.N*self.nu + self.car.N+1 + 2*self.K_NEAR + i, 
                                 (self.car.N+1)*self.nx + self.car.N*self.nu + self.car.N+1 + 2*self.K_NEAR + i, self.car.q_s_terminal)
        
        lower[:self.nx] = -x0
        upper[:self.nx] = -x0

        H_t = HessianMatrix.transpose()
        sparse_I = sparse.eye((self.car.N+1)*self.nx + self.car.N*self.nu + self.car.N+1 + 2*self.K_NEAR + self.nx)

        HessianMatrix = 0.5 * (HessianMatrix + H_t) + 1e-7 * sparse_I
        
        self.osqp.update(P=HessianMatrix, q=gradient, A=constraintMatrix, l=lower, u=upper)
        results = self.osqp.solve()

        self.QPSol = results.x

        

    def apply_control(self):
        # line 938: apply_control
        accel = self.QPSol[(self.car.N+1)*self.nx]
        steer = self.QPSol[(self.car.N+1)*self.nx + 1]

        self.get_logger().info(f"accel_cmd: {accel}, steer_cmd: {steer}, slip_angle: {self.slip_angle}")
        steer = np.clip(steer, -self.car.steer_max, self.car.steer_max)

        drive_msg = AckermannDriveStamped()
        drive_msg.header.stamp = self.get_clock().now().to_msg()
        drive_msg.header.frame_id = 'base_link'
        drive_msg.drive.steering_angle = steer
        drive_msg.drive.steering_angle_velocity = 1.0
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