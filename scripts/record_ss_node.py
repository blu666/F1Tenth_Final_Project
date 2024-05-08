#!/usr/bin/env python3
import sys

import rclpy.time
sys.path.append("./")
import rclpy
from rclpy.node import Node
import numpy as np
from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry
from scipy.spatial.transform import Rotation as R
from nav_msgs.msg import OccupancyGrid
from visualization_msgs.msg import Marker, MarkerArray
from utils.track import Track
from utils.spline import Spline
from builtin_interfaces.msg import Time
from copy import deepcopy


class RecordSS(Node):
    """ 
    Implement Pure Pursuit on the car
    This is just a template, you are free to implement your own node!
    """
    def __init__(self):
        super().__init__('record_ss_node')
        self.savepath = "./map/initial_ss.csv"

        ## Subs
        self.create_subscription(Odometry, '/ego_racecar/odom', self.pose_callback, 10)
        # self.create_subscription(Odometry, '/pf/pose/odom', self.pose_callback, 10)
        self.create_subscription(AckermannDriveStamped, '/drive', self.control_callback, 10)
        self.create_subscription(OccupancyGrid, '/map', self.map_callback, 10)
        self.goalpoint_publisher = self.create_publisher(Marker, '/pure_pursuit/goalpoint', 10)
        self.waypoints_publisher = self.create_publisher(MarkerArray, '/pure_pursuit/waypoints', 10)
        self.track = Track("./map/race3/centerline_new.csv", "./map/race3/race3_track.png")
        self.reset_startingline = True
        self.u = np.zeros(2, dtype=np.float32)
        # self.x = np.zeros(6, dtype=np.float32)
        self.time = 0
        self.lap = 1
        self.odom: Odometry = None
        self.prev_odom: Odometry = None
        # self.vx = 0.0
        # self.vy = 0.0
        # self.prev_position = 0.0
        # self.prev_time = None
        # self.X = 0.0  
        # self.Y = 0.0  
        # self.yaw = 0.0    
        # print("init time", self.prev_time, self.prev_time.sec, self.prev_time.nanosec )
        # print(self.prev_time.to_msg(), self.prev_time.to_msg().sec, self.prev_time.to_msg().nanosec)
        self.map: OccupancyGrid = None
        self.s_prev = 0
        self.record = []
        self.start_record = False
        self.is_finished = False
        self.waypoints = None
        # self.is_initized = False

    def check_init(self):
        if self.odom is None or self.map is None:
            return False

    def control_callback(self, drive_msg: AckermannDriveStamped):
        # return
        if drive_msg is None:
            self.get_logger().info("@=> INVALID DRIVE MESSAGE")
            return
        if self.odom is None:
            self.get_logger().info("@=> Waiting for odometry message")
            return
        if self.prev_odom is None:
            self.get_logger().info("@=> Waiting for previous odometry message")
            self.prev_odom = self.odom
            return
        if self.is_finished:
            return
        ## Extract states from odometry and drive message
        current_odom = deepcopy(self.odom)
        accel = drive_msg.drive.acceleration
        
        self.u = np.array([drive_msg.drive.steering_angle, accel], dtype=np.float32) # [delta, a]
        
        ## TO GLOBAL
        X, Y = current_odom.pose.pose.position.x, current_odom.pose.pose.position.y
        # elapsed_time = current_odom.header.stamp.sec - self.prev_odom.header.stamp.sec + \
        #     (current_odom.header.stamp.nanosec - self.prev_odom.header.stamp.nanosec) * 1e-9
        # vx_glob = (X - self.prev_odom.pose.pose.position.x) / elapsed_time
        # vy_glob = (Y - self.prev_odom.pose.pose.position.y) / elapsed_time
        # print(vx_glob, vy_glob)
        heading = R.from_quat(np.array([current_odom.pose.pose.orientation.x,
                                        current_odom.pose.pose.orientation.y,
                                        current_odom.pose.pose.orientation.z,
                                        current_odom.pose.pose.orientation.w]))
        yaw = heading.as_euler('zyx')[0]
        # vx, vy, _ = heading.inv().apply([vx_glob, vy_glob, 0])
        vx = current_odom.twist.twist.linear.x
        vy = np.random.randn() * 1e-3
        # print(odom_vx, vx, vy)
        wz = current_odom.twist.twist.angular.z
        ## Extract states from track
        epsi, s_curr, ey, closepoint = self.track.get_states(X, Y, yaw)
        self.publish_goalpoint(closepoint)
        # self.get_logger().info("v:({0:.4f},{1:.4f}), odom: {2:.4f}, v_glob: {3:.4f}".format(vx, vy, odom_vx, np.linalg.norm([vx_glob, vy_glob])))
        ## Check if the car has passed the starting line
        if not self.start_record:
            if s_curr < self.track.length / 3:
                self.start_record = True
            else:
                return
        if s_curr - self.s_prev < -self.track.length / 2:
            self.time = 0
            self.lap += 1
            print("!===============STARTING NEW LAP {}".format(s_curr - self.s_prev))
            if self.lap > 10:
                # Stop recording after two laps
                self.get_logger().info("Finished recording")
                self.save_record(self.savepath)
                self.is_finished = True
                rclpy.shutdown()
        self.s_prev = s_curr
        # xPID_cl is [vx, vy, wz, epsi, s, ey]; xPID_cl_glob is [vx, vy, wz, psi, X, Y]
        self.record.append([self.time, self.lap, vx, vy, wz, epsi, s_curr, ey, yaw, X, Y, self.u[0], self.u[1]])
        print("Recorded lap {}, time {}, s {}".format(self.lap, self.time, s_curr))
        # print(self.time, self.lap, epsi, s_curr, ey)
        
        self.time += 1
        self.prev_odom = current_odom

    def save_record(self, savepath: str):
        header = "self.time, self.lap, vx, vy, wz, epsi, s_curr, ey, yaw, X, Y, self.u[0], self.u[1]"
        np.savetxt(savepath, 
                   self.record, 
                   fmt='%1.6f,%1.6f,%1.6f,%1.6f,%1.6f,%1.6f,%1.6f,%1.6f,%1.6f,%1.6f,%1.6f,%1.6f,%1.6f',
                    header=header
        )
        self.get_logger().info("Initial SS saved to {}".format(savepath))
        return    

    def pose_callback(self, pose_msg: Odometry):
        # t = pose_msg.header.stamp
        self.odom = pose_msg
        
        if self.reset_startingline:
            self.get_logger().info("Reset starting line")
            self.track.reset_starting_point_new(pose_msg.pose.pose.position.x,
                                            pose_msg.pose.pose.position.y,
                                            refine=True)
            self.waypoints = self.track.centerline_xy
            self.publish_waypoints()
            self.reset_startingline = False
            return
        
        # self.u = np.array([drive_msg.drive.steering_angle, accel], dtype=np.float32) # [delta, a]
        ##===== For debugging
        # X, Y = self.odom.pose.pose.position.x, self.odom.pose.pose.position.y
        # heading = R.from_quat(np.array([self.odom.pose.pose.orientation.x,
        #                                 self.odom.pose.pose.orientation.y,
        #                                 self.odom.pose.pose.orientation.z,
        #                                 self.odom.pose.pose.orientation.w]))
        # yaw = heading.as_euler('zyx')[0]
        # vx = self.odom.twist.twist.linear.x
        # vy = self.odom.twist.twist.linear.y
        # wz = self.odom.twist.twist.angular.z
        
        # ## Extract states from track
        # # epsi, s_curr, ey = self.track.get_states(X, Y, yaw)
        # epsi, s_curr, ey, closepoint = self.track.get_states(X, Y, yaw)
        
        # xc, yc, yawc = self.track.track_to_global(ey, epsi, s_curr)
        # x_proj, y_proj = self.track.x_eval(s_curr), self.track.y_eval(s_curr)
        # self.publish_goalpoint(closepoint)
        # self.get_logger().info("e_y {} s {} e_yaw {}".format(ey, s_curr, epsi))
        # self.get_logger().info("xy {},{} rxy {},{}, err:{}".format(X, Y, xc, yc, np.linalg.norm([X - xc, Y - yc])))
        # self.get_logger().info("yaw {}, ryaw{} err:{}".format(yaw, yawc, np.abs(yaw - yawc)))
        # self.get_logger().info("close {}, {}".format(closepoint, (x_proj, y_proj)))
    
    def map_callback(self, map_msg: OccupancyGrid):
        self.get_logger().info("Map received")
        return

    def publish_waypoints(self):
        if self.waypoints is None:
            print("WAYPOINTS NOT INITIALIZED")
            return
        print("INITIALIZE WAYPOINTS PUBLISHER")
        markerArray = MarkerArray()
        for i, wp in enumerate(self.waypoints):
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

    def publish_goalpoint(self, goalpoint):
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.id = 0
        marker.type = marker.SPHERE
        marker.action = marker.ADD
        marker.pose.position.x = goalpoint[0]
        marker.pose.position.y = goalpoint[1]
        marker.pose.position.z = 0.0
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.2
        marker.scale.y = 0.2
        marker.scale.z = 0.2
        marker.color.a = 1.0
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        self.goalpoint_publisher.publish(marker)


def main(args=None):
    rclpy.init(args=args)
    print("Initialize Safty Set Node")
    record_ss_node = RecordSS()
    rclpy.spin(record_ss_node)
    record_ss_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
