#!/usr/bin/env python3
import sys
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

class RecordSS(Node):
    """ 
    Implement Pure Pursuit on the car
    This is just a template, you are free to implement your own node!
    """
    def __init__(self):
        super().__init__('record_ss_node')
        self.savepath = "./map/inital_ss.csv"
        ## Subs
        self.create_subscription(Odometry, '/ego_racecar/odom', self.pose_callback, 10)
        # self.create_subscription(Odometry, '/pf/pose/odom', self.pose_callback, 10)
        self.create_subscription(AckermannDriveStamped, '/drive', self.control_callback, 10)
        self.create_subscription(OccupancyGrid, '/map', self.map_callback, 10)
        self.goalpoint_publisher = self.create_publisher(Marker, '/pure_pursuit/goalpoint', 10)
        self.waypoints_publisher = self.create_publisher(MarkerArray, '/pure_pursuit/waypoints', 10)
        
        self.track = Track("./map/levine/centerline.csv")
        self.reset_startingline = True
        
        self.u = np.zeros(2, dtype=np.float32)
        # self.x = np.zeros(6, dtype=np.float32)
        
        self.time = 0
        self.lap = 0
        self.odom: Odometry = None
        self.map: OccupancyGrid = None
        self.s_prev = 0
        self.record = []
        self.is_finished = False
        self.waypoints = None
        # self.is_initized = False
    
    def check_init(self):
        if self.odom is None or self.map is None:
            return False
    
    def control_callback(self, drive_msg: AckermannDriveStamped):
        # return
        if drive_msg is None:
            self.get_logger().info("INVALID DRIVE MESSAGE")
            return
        if self.odom is None:
            self.get_logger().info("Waiting for odometry message")
            return
        if self.is_finished:
            return
        self.get_logger().info("Recording lap: {} at time: {}".format(self.lap, self.time))
        ## Extract states from odometry and drive message
        accel = drive_msg.drive.acceleration
        self.u = np.array([drive_msg.drive.steering_angle, accel], dtype=np.float32) # [delta, a]
        X, Y = self.odom.pose.pose.position.x, self.odom.pose.pose.position.y
        heading = R.from_quat(np.array([self.odom.pose.pose.orientation.x,
                                        self.odom.pose.pose.orientation.y,
                                        self.odom.pose.pose.orientation.z,
                                        self.odom.pose.pose.orientation.w]))
        yaw = heading.as_euler('zyx')[0]
        vx = self.odom.twist.twist.linear.x
        vy = self.odom.twist.twist.linear.y
        wz = self.odom.twist.twist.angular.z
        
        ## Extract states from track
        epsi, s_curr, ey, closepoint = self.track.get_states(X, Y, yaw)
        self.publish_goalpoint(closepoint)
        ## Check if the car has passed the starting line
        if s_curr - self.s_prev < -self.track.length / 2:
            self.time = 0
            self.lap += 1
            if self.lap > 1:
                # Stop recording after two laps
                self.get_logger().info("Finished recording")
                self.save_record(self.savepath)
                self.is_finished = True
                rclpy.shutdown()
        self.s_prev = s_curr
        # xPID_cl is [vx, vy, wz, epsi, s, ey]; xPID_cl_glob is [vx, vy, wz, psi, X, Y]
        self.record.append([self.time, self.lap, vx, vy, wz, epsi, s_curr, ey, yaw, X, Y, self.u[0], self.u[1]])
        print(self.time, self.lap, epsi, s_curr, ey)
        self.time += 1

    def save_record(self, savepath: str):
        # header="time, x, y, yaw, vel, acc_cmd, steer_cmd, s, lap"
        np.savetxt(savepath, self.record, delimiter=",")
        self.get_logger().info("Initial SS saved to {}".format(savepath))
        return    

    def pose_callback(self, pose_msg: Odometry):
        # t0 = Time.from_msg(pose_msg.header.stamp)
        self.odom = pose_msg
        if self.reset_startingline:
            self.get_logger().info("Reset starting line")
            self.track.reset_starting_point(pose_msg.pose.pose.position.x,
                                            pose_msg.pose.pose.position.y,
                                            refine=True)
            self.waypoints = self.track.centerline_xy
            self.publish_waypoints()
            self.reset_startingline = False
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
        # self.publish_goalpoint(closepoint)
        # print(X, Y, epsi, s_curr, ey)
        
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
