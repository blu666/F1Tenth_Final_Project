#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.time import Time
import numpy as np
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped, AckermannDrive
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker, MarkerArray
from scipy.spatial.transform import Rotation as R
# from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy
from time import sleep
from track import Track

class RecordSS(Node):
    """ 
    Implement Pure Pursuit on the car
    This is just a template, you are free to implement your own node!
    """
    def __init__(self):
        super().__init__('record_ss_node')
        
        # self.create_subscription(Odometry, '/ego_racecar/odom', self.pose_callback, 10)
        self.create_subscription(Odometry, '/pf/pose/odom', self.pose_callback, 10)
        self.create_subscription(AckermannDriveStamped, '/drive', self.accel_callback, 10)
        self.track = Track("./map/levine/centerline.csv") # TODO: generate centerline.csv

    def accel_callback(self, drive_msg: AckermannDriveStamped):
        
        return
    
    def pose_callback(self, pose_msg: Odometry):
        # t0 = Time.from_msg(pose_msg.header.stamp)
        current_pos = np.array([pose_msg.pose.pose.position.x, pose_msg.pose.pose.position.y])
        current_heading = R.from_quat(np.array([pose_msg.pose.pose.orientation.x, pose_msg.pose.pose.orientation.y, pose_msg.pose.pose.orientation.z, pose_msg.pose.pose.orientation.w]))
        
        
        # find current waypoint by projecting the car forward by lookahead distance, then finding the closest waypoint to that projected position
        # depending on the distance of the closest waypoint to current position, we will find two waypoints that sandwich the current position plus lookahead distance
        # then we interpolate between these two waypoints to find the current waypoint
        current_waypoint, current_params = self.find_current_waypoint(current_pos, current_heading)
        self.publish_goalpoint(current_waypoint)
    
        # transform the current waypoint to the vehicle frame of reference
        self.map_to_car_translation = np.array([pose_msg.pose.pose.position.x, pose_msg.pose.pose.position.y, pose_msg.pose.pose.position.z])
        self.map_to_car_rotation = R.from_quat([pose_msg.pose.pose.orientation.x, pose_msg.pose.pose.orientation.y, pose_msg.pose.pose.orientation.z, pose_msg.pose.pose.orientation.w])

        
        wp_car_frame = (np.array([current_waypoint[0], current_waypoint[1], 0]) - self.map_to_car_translation)
        wp_car_frame = wp_car_frame @ self.map_to_car_rotation.as_matrix()

        self.lookahead = current_params[2] 
        curvature = 2 * wp_car_frame[1] / self.lookahead**2
        
        drive_msg = AckermannDriveStamped()
        drive_msg.header.stamp = self.get_clock().now().to_msg()
        drive_msg.header.frame_id = "ego_racecar/base_link"
        drive_msg.drive.steering_angle = current_params[3] * curvature + current_params[4] * (self.last_curve - curvature)
        pf_speed = np.linalg.norm(np.array([pose_msg.twist.twist.linear.x, pose_msg.twist.twist.linear.y]))
        drive_msg.drive.speed = self.interpolate_vel(pf_speed, current_params[0] * current_params[1])
        self.get_logger().info("pf speed: {} seg speed: {} command: {}".format(pf_speed, current_params[0] * current_params[1], drive_msg.drive.speed))
        self.drive_publisher.publish(drive_msg)
        # t1 = Time.from_msg(drive_msg.header.stamp)
        # self.get_logger().info("Time taken: {}".format((t1 - t0).nanoseconds / 1e9))


    ### Visulization functions
    def publish_waypoints(self):
        if len(self.waypoints) == 0:
            return
        
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
            marker.scale.x = 0.2
            marker.scale.y = 0.2
            marker.scale.z = 0.2
            marker.color.a = 1.0
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.0
            markerArray.markers.append(marker)
        self.waypoints_publisher.publish(markerArray)

    def publish_future_pos(self, future_pos):
        if not self.vis:
            return
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.id = 0
        marker.type = marker.SPHERE
        marker.action = marker.ADD
        marker.pose.position.x = future_pos[0]
        marker.pose.position.y = future_pos[1]
        marker.pose.position.z = 0.0
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.25
        marker.scale.y = 0.25
        marker.scale.z = 0.25
        marker.color.a = 1.0
        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 1.0
        self.future_pos_publisher.publish(marker)

    def publish_testpoints(self, testpoints):
        if not self.vis:
            return
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
            marker.color.g = 0.0
            marker.color.b = 1.0
            markerArray.markers.append(marker)
        self.testpoint_publisher.publish(markerArray)


    def publish_goalpoint(self, goalpoint):
        if not self.vis:
            return
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
    pure_pursuit_node = RecordSS()
    rclpy.spin(pure_pursuit_node)

    pure_pursuit_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
