#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np
from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry
from scipy.spatial.transform import Rotation as R
from track import Track


class RecordSS(Node):
    """ 
    Implement Pure Pursuit on the car
    This is just a template, you are free to implement your own node!
    """
    def __init__(self):
        super().__init__('record_ss_node')
        self.savepath = "./map/inital_ss.csv"
        self.create_subscription(Odometry, '/ego_racecar/odom', self.pose_callback, 10)
        # self.create_subscription(Odometry, '/pf/pose/odom', self.pose_callback, 10)
        self.create_subscription(AckermannDriveStamped, '/drive', self.control_callback, 10)
        
        self.track = Track("./map/levine/centerline.csv") # TODO: generate centerline.csv
        self.u = np.zeros(2, dtype=np.float32)
        self.x = np.zeros(6, dtype=np.float32)
        self.time = 0
        self.lap = 0
        self.odom: Odometry = None
        self.s_prev = 0
        self.record = []
        self.is_finished = False
        self.is_startingline_reset = False

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
        self.u = np.array([drive_msg.drive.speed, drive_msg.drive.steering_angle], dtype=np.float32)
        pos = np.array([self.odom.pose.pose.position.x,
                    self.odom.pose.pose.position.y])
        heading = R.from_quat(np.array([self.odom.pose.pose.orientation.x,
                                        self.odom.pose.pose.orientation.y,
                                        self.odom.pose.pose.orientation.z,
                                        self.odom.pose.pose.orientation.w]))
        # tf2_py.transformations.euler_from_quaternion
        yaw = heading.as_euler('zyx')[0]
        # yaw_dot = self.odom.twist.twist.angular.z
        vel = self.odom.twist.twist.linear.x
        s_curr = self.track.get_theta(pos[0], pos[1])
        if s_curr - self.s_prev < -self.track.length / 2:
            self.time = 0
            self.lap += 1
            # TODO: Stop recording after two laps
            if self.lap > 1:
                self.get_logger().info("Finished recording")
                # self.save_record(self.savepath)
                self.is_finished = True
                rclpy.shutdown()
        self.s_prev = s_curr
        self.record.append([self.time, pos[0], pos[1], yaw, vel, self.u[0], self.u[1], s_curr, self.lap])
        print(self.time, pos[0], pos[1], yaw, vel, self.u[0], self.u[1], s_curr, self.lap)
        self.time += 1

    def save_record(self, savepath: str):
        np.savetxt(savepath, self.record, delimiter=",", header="##time, x, y, yaw, vel, speed_cmd, steer_cmd, s, lap")
        self.get_logger().info("Initial SS saved to {}".format(savepath))
        return    

    def pose_callback(self, pose_msg: Odometry):
        # t0 = Time.from_msg(pose_msg.header.stamp)
        self.odom = pose_msg
        if not self.is_startingline_reset:
            self.get_logger().info("Reset starting line")
            self.track.reset_starting_point(pose_msg.pose.pose.position.x, pose_msg.pose.pose.position.y)
            self.is_startingline_reset = True


def main(args=None):
    rclpy.init(args=args)
    print("Initialize Safty Set Node")
    record_ss_node = RecordSS()
    rclpy.spin(record_ss_node)
    record_ss_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
