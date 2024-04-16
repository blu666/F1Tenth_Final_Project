#!/usr/bin/env python3
"""
This file contains the class definition for tree nodes and RRT
Before you start, please read: https://arxiv.org/pdf/1105.1186.pdf
"""
import numpy as np
from numpy import linalg as LA
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
from matplotlib import pyplot as plt
from copy import deepcopy
from scipy.spatial.transform import Rotation as R
import tf2_ros
from visualization_msgs.msg import Marker, MarkerArray
from rclpy.parameter import Parameter, ParameterType
from sensor_msgs.msg import PointCloud
from occupancy_grid import CarOccupancyGrid
# from rrt_star import RRT_Star


# class def for RRT
class LMPC(Node):
    def __init__(self):
        super().__init__('lmpc_node')
        # topics, not saved as attributes
        
        ## topics
        self.declare_parameter("pose_topic", "/ego_racecar/odom")
        self.declare_parameter("scan_topic", "/scan")
        ## map update rate
        self.declare_parameter("map_rate", 25.0)
        ## control parameters
        self.declare_parameter("vel", 0.8)
        self.declare_parameter("p", 0.15)
        self.declare_parameter("d", 0.8)
        self.declare_parameter("lookahead",3.5)
        ## visualization ON/OFF
        self.declare_parameter("visualize", True)
        ## RRT parameters
        self.declare_parameter("rrt_lookahead", 0.7)
        self.declare_parameter("rewire_radius", 12.0)
        self.declare_parameter("steer_distance", 8.0)
        self.declare_parameter("goal_threshold", 10.0)
        
        pose_topic = self.get_parameter("pose_topic").get_parameter_value().string_value
        scan_topic = self.get_parameter("scan_topic").get_parameter_value().string_value
        self.lookahead = self.get_parameter("lookahead").get_parameter_value().double_value
        self.rrt_lookahead = self.get_parameter("rrt_lookahead").get_parameter_value().double_value
        self.map_rate = self.get_parameter("map_rate").get_parameter_value().double_value
        self.vel = self.get_parameter("vel").get_parameter_value().double_value
        self.p = self.get_parameter("p").get_parameter_value().double_value
        self.visualize = self.get_parameter("visualize").get_parameter_value().bool_value
        self.steer_distance = self.get_parameter("steer_distance").get_parameter_value().double_value
        self.goal_threshold = self.get_parameter("goal_threshold").get_parameter_value().double_value
        self.rewire_radius = self.get_parameter("rewire_radius").get_parameter_value().double_value
        
        ## Initialized in callbacks
        self.scan_params = None
        self.goal_point_cf = None
        self.goal_point = None
        self.prev_scan_stamp = None
        rrt_wp = None
        self.map_to_car_transform = None
        
        # you could add your own parameters to the rrt_params.yaml file,
        # and get them here as class attributes as shown above.
        
        self.pose_sub_ = self.create_subscription(
            Odometry,
            pose_topic,
            self.pose_callback,
            1)

        self.scan_sub_ = self.create_subscription(
            LaserScan,
            scan_topic,
            self.scan_callback,
            1)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        # publishers
        self.drive_pub_ = self.create_publisher(
            AckermannDriveStamped,
            "/drive",
            1)
        self.waypoints = self.load_waypoints("waypoints/wp_interpolated.csv")
        
        # Visualization publishers
        if self.visualize:
            self.waypoints_publisher = self.create_publisher(MarkerArray, '/rrt/waypoints', 50)
            self.rrtpoint_publisher = self.create_publisher(Marker, '/rrt/rrtpoint', 5)
            self.goalpoint_publisher = self.create_publisher(Marker, '/rrt/goalpoint', 5)
            self.path_publisher = self.create_publisher(Marker, '/rrt/path', 10)
            self.tree_publisher = self.create_publisher(MarkerArray, '/rrt/tree', 10)
            self.leaf_nodes_publisher = self.create_publisher(MarkerArray, '/rrt/leaf_nodes', 10)
            self.occupancy_grid_publisher = self.create_publisher(MarkerArray, '/rrt/occupancy_grid', 10)
            self.publish_waypoints()

        # Occupancy grid
        self.occupancy_grid = CarOccupancyGrid(width=2.5,    # MODIFY OCCUPENCY MAP SIZE HERE
                                            height=4.5,   # MODIFY OCCUPENCY MAP SIZE HERE
                                            resolution=(0.05, 0.05),    # MODIFY OCCUPENCY MAP RESOLUTION HERE
                                            margin=0.2)     # MODIFY OCCUPENCY MAP MARGIN HERE
        self.get_logger().info("RRT Node has been initialized")
        

    def scan_callback(self, scan_msg: LaserScan):
        """
        LaserScan callback, you should update your occupancy grid here

        Args: 
            scan_msg (LaserScan): incoming message from subscribed topic
        Returns:

        """
        if self.scan_params is None:
            self.scan_params = {"angle_min": scan_msg.angle_min, 
                                "angle_max": scan_msg.angle_max, 
                                "angle_increment": scan_msg.angle_increment, 
                                "range_min": scan_msg.range_min, 
                                "range_max": scan_msg.range_max}
        time_stamp = scan_msg.header.stamp
        if self.prev_scan_stamp is not None:
            if (time_stamp.sec + time_stamp.nanosec * 1e-9)\
                - (self.prev_scan_stamp.sec + self.prev_scan_stamp.nanosec * 1e-9)\
                    < 1/self.map_rate:
                return
            
        self.prev_scan_stamp = time_stamp
        self.min_angle_idx = int((np.deg2rad(-90) - self.scan_params["angle_min"]) / self.scan_params["angle_increment"])
        self.max_angle_idx = int((np.deg2rad(90) - self.scan_params["angle_min"]) / self.scan_params["angle_increment"])
    
        ranges = np.array(scan_msg.ranges[self.min_angle_idx:self.max_angle_idx+1])
        # ranges = np.clip(ranges, 0, 10)
        ## coordinate in car frame
        x_cf = np.cos(np.arange(np.deg2rad(-90), np.deg2rad(90), self.scan_params["angle_increment"])) * np.array(ranges)
        y_cf = np.sin(np.arange(np.deg2rad(-90), np.deg2rad(90), self.scan_params["angle_increment"])) * np.array(ranges)
        self.occupancy_grid.clear()
        self.occupancy_grid.update_scan(x_cf, y_cf)
        # self.occupancy_grid.display_map()
        ## publish coordinates of occupied cells in world frame
        if self.visualize:
            if self.map_to_car_transform is None:
                return
            map_to_car_rotation = self.map_to_car_rotation
            map_to_car_translation = self.map_to_car_translation
            xs_cf, ys_cf = self.occupancy_grid.occpency_in_cf()
            ps_wf = np.vstack((xs_cf, ys_cf, np.zeros(xs_cf.shape[0]))).T
            ps_wf = ps_wf @ map_to_car_rotation.as_matrix().T + map_to_car_translation
            self.publish_occupancy_grid(ps_wf)
        return

    def pose_callback(self, pose_msg: Odometry):
        """
        The pose callback when subscribed to particle filter's inferred pose
        Here is where the main RRT loop happens

        Args: 
            pose_msg (PoseStamped): incoming message from subscribed topic
        Returns:

        """
        
        # 1. find goal point
        current_pos = np.array([pose_msg.pose.pose.position.x, pose_msg.pose.pose.position.y])
        current_heading = R.from_quat(np.array([pose_msg.pose.pose.orientation.x, pose_msg.pose.pose.orientation.y, pose_msg.pose.pose.orientation.z, pose_msg.pose.pose.orientation.w]))
        current_waypoint = self.find_current_waypoint(current_pos, current_heading)
        # self.publish_goalpoint(current_waypoint)
        try:
            self.tf_buffer.can_transform("map", "ego_racecar/base_link", rclpy.time.Time(), rclpy.duration.Duration(seconds=0.1))
            self.map_to_car_transform = self.tf_buffer.lookup_transform("map", "ego_racecar/base_link", rclpy.time.Time())
        except:
            print("No transform found")
            return
        map_to_car_translation = self.map_to_car_transform.transform.translation
        self.map_to_car_translation = np.array([map_to_car_translation.x, map_to_car_translation.y, map_to_car_translation.z])
        map_to_car_rotation = self.map_to_car_transform.transform.rotation
        self.map_to_car_rotation = R.from_quat([map_to_car_rotation.x, map_to_car_rotation.y, map_to_car_rotation.z, map_to_car_rotation.w])
        
        wp_car_frame = (np.array([current_waypoint[0], current_waypoint[1], 0]) - self.map_to_car_translation)
        wp_car_frame = wp_car_frame @ self.map_to_car_rotation.as_matrix()
        

        # print("Waypoint in car frame: ", wp_car_frame)

        # 2. run RRT
        x, y = self.occupancy_grid.cf_to_idx(wp_car_frame[0], wp_car_frame[1])
        rrt_goal_point = [x[0], y[0]] ### GOAL POINT FOR RRT
        self.rrt_star = RRT_Star(occupancy_grid = deepcopy(self.occupancy_grid.occupancy_grid),
                                 start = np.array(self.occupancy_grid.origin), 
                                 goal = rrt_goal_point,
                                 max_iter=400,
                                 steer_distance=self.steer_distance,
                                 goal_threshold=self.goal_threshold,
                                 rewire_radius=self.rewire_radius)


        self.occupancy_grid_cp = deepcopy(self.occupancy_grid.occupancy_grid)
        self.goal_point = wp_car_frame
        self.publish_goalpoint([current_waypoint[0], current_waypoint[1]])
        # self.occupancy_grid.update_cell(wp_car_frame[1], wp_car_frame[0], 3)
        self.goal_point_cf = wp_car_frame
        
        # t0 = self.get_clock().now()
        self.path = self.rrt_star.RRTStar() # returns a list of coordinates in car frame
        # t1 = self.get_clock().now()
        # print("Time taken for RRT: ", (t1 - t0).nanoseconds * 1e-9)
        self.tree = self.rrt_star.pts[:self.rrt_star.curr_num, :]
        # print(self.tree.shape)
        self.publish_path()
        self.publish_tree()
        self.publish_leaf_nodes()
        # print("path found", self.path)
        if self.path is None or len(self.path) <= 1:
            if self.path is None:
                self.get_logger().info("path is None")
            else:
                self.get_logger().info("path less than 2")
                # print(self.path)
                # for idx in self.path:
                #     self.get_logger().info("node: " + str(self.path[idx][0]) + " " + str(self.path[idx][1]))
            return
        rrt_wp = self.find_rrt_waypoint()
        t1 = self.get_clock().now()
        # print("Time taken for RRT: ", (t1 - t0).nanoseconds * 1e-9)
        self.tree = self.rrt_star.pts[:self.rrt_star.curr_num, :]
        # print(self.tree.shape)
        self.publish_path()
        self.publish_tree()
        self.publish_leaf_nodes()
        # print("path found", self.path)
        
        
        # rrt_wp = [self.path[1][0], self.path[1][1]]
        # rrt_wp = [-(rrt_wp[0] - self.occupancy_grid.origin[0]) * self.occupancy_grid.resolution[0], 
        #                (rrt_wp[1] - self.occupancy_grid.origin[1]) * self.occupancy_grid.resolution[1]]

        # rrt_wp_world_frame = np.array([rrt_wp[1], rrt_wp[0], 0]) @ self.map_to_car_rotation.as_matrix().T + self.map_to_car_translation
        # self.publish_rrtpoint([rrt_wp_world_frame[0], rrt_wp_world_frame[1]])
        # self.get_logger().info("rrt_wp: " + str(rrt_wp[0]) + " " + str(rrt_wp[1]))
        y_err = rrt_wp[0]
        # self.get_logger().info("y_err: " + str(y_err))
        # new_lookahead = LA.norm(np.array([rrt_wp[0], rrt_wp[1]]))
        # self.get_logger().info("new_lookahead: " + str(new_lookahead))
        curvature = 2 * y_err / ((LA.norm(rrt_wp) - 0.1 * self.vel)**2)
        # curvature = 2 * y_err / ((self.rrt_lookahead - 0.1 * self.vel)**2)
        # self.get_logger().info("curvature: " + str(curvature))
        
        drive_msg = AckermannDriveStamped()
        drive_msg.header.stamp = self.get_clock().now().to_msg()
        drive_msg.header.frame_id = "map"
        drive_msg.drive.steering_angle = np.clip(self.p * curvature, np.deg2rad(-60), np.deg2rad(60))
        # self.get_logger().info("steering angle: " + str(drive_msg.drive.steering_angle))
        drive_msg.drive.speed = self.vel
        # if abs(drive_msg.drive.steering_angle) > np.deg2rad(30):
        #     drive_msg.drive.speed = 0.5
        self.drive_pub_.publish(drive_msg)


    def find_rrt_waypoint(self):
        path_car_frame = (self.path - self.occupancy_grid.origin) * self.occupancy_grid.resolution * np.array([-1, 1])
        # print(self.path)
        # print(path_car_frame)
        wp_dist = np.abs(LA.norm(path_car_frame, axis=1) - self.rrt_lookahead)
        target_idx = np.argmin(wp_dist)
        if wp_dist[target_idx] < self.rrt_lookahead:
            two_wps = path_car_frame[target_idx:target_idx+2]
        else:
            two_wps = path_car_frame[target_idx-1:target_idx+1]
        # print(path_car_frame)
        # print(two_wps)
        # print(two_wps[0])
        # print(path_car_frame.shape, target_idx)
        # print(two_wps)
        if two_wps.shape[0] == 1:
            target_wp = two_wps[0]
        else:
            target_wp = self.interpolate_waypoints(two_wps, np.array([0, 0]), self.rrt_lookahead)
        rrt_wp_world_frame = np.array([target_wp[1], target_wp[0], 0]) @ self.map_to_car_rotation.as_matrix().T + self.map_to_car_translation
        self.publish_rrtpoint([rrt_wp_world_frame[0], rrt_wp_world_frame[1]])
        # print(target_wp)
        return target_wp


    def load_waypoints(self, path):
        waypoints = []
        with open(path, newline='') as f:
            reader = csv.reader(f)
            waypoints = list(reader)
            waypoints = [np.array([float(wp[0]), float(wp[1])]) for wp in waypoints]
        
        return waypoints
    
    def find_current_waypoint(self, current_pos, current_heading):
        euler_angles = current_heading.as_euler('zyx')
        future_pos = current_pos + self.lookahead * np.array([np.cos(euler_angles[0]), np.sin(euler_angles[0])])
        closest_wp = None
        min_dist = float('inf')
        for idx, wp in enumerate(self.waypoints):
            dist = np.linalg.norm(np.array(wp) - future_pos)
            if dist < min_dist:
                min_dist = dist
                closest_wp = wp
                min_idx = idx
        
        # self.publish_testpoints([closest_wp])
        dist_to_curr_pos = np.linalg.norm(closest_wp - current_pos)
        if dist_to_curr_pos <= self.lookahead:
            two_wps = [self.waypoints[min_idx]]
            if min_idx+1 < len(self.waypoints):
                two_wps.append(self.waypoints[min_idx+1])
            else:
                two_wps.append(self.waypoints[0])
        else:
            two_wps = []
            if min_idx-1 >= 0:
                two_wps.append(self.waypoints[min_idx-1])
            else:
                two_wps.append(self.waypoints[-1])
            two_wps.append(self.waypoints[min_idx])
        # print(two_wps, future_pos)
        # print(future_pos, two_wps)
        # self.publish_future_pos(future_pos)
        # self.publish_testpoints(two_wps)
        return self.interpolate_waypoints(two_wps, current_pos, self.lookahead)

    def interpolate_waypoints(self, two_wps, curr_pos, lookahead):
        # print(two_wps)
        # self.publish_testpoints(two_wps)

        wp_vec = two_wps[0] - two_wps[1]
        pos_vec = two_wps[0] - curr_pos
        alpha = np.arccos(np.dot(wp_vec, pos_vec) / (np.linalg.norm(wp_vec) * np.linalg.norm(pos_vec)))
        beta = np.pi - alpha
        a = np.linalg.norm(pos_vec) * np.cos(beta)
        b = np.linalg.norm(pos_vec) * np.sin(beta)
        c = np.sqrt(lookahead**2 - b**2) - a
        return two_wps[0] - c * wp_vec / np.linalg.norm(wp_vec)

    ###### Vis code, comment after finishing

    def find_leaf_nodes(self):
        # Initialize a set to keep track of indices of nodes that are parents
        parent_indices = set()

        # Populate the set with indices of all parent nodes
        for idx in self.rrt_star.parents:
            if idx != -2:  # Assuming None or a similar placeholder for root
                parent_indices.add(idx)
        
        # Identify leaf nodes
        # A leaf node is a node that is not a parent to any other node
        leaf_nodes = []
        for idx, node in enumerate(self.tree):
            if idx not in parent_indices:
                # This node is not a parent of any node, hence it's a leaf node
                leaf_nodes.append(node)
        
        return leaf_nodes

    def publish_leaf_nodes(self):
        if not self.visualize:
            return
        marker_array = MarkerArray()
        leaf_nodes = self.find_leaf_nodes()

        for idx, node in enumerate(leaf_nodes):
            node_global = self.occupancy_grid.omap_to_global(node, self.map_to_car_rotation, self.map_to_car_translation)
            marker = Marker()
            marker.header.frame_id = "map"  # Adjust based on your TF frames
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "rrt_leaf_nodes"
            marker.id = idx
            marker.type = Marker.SPHERE  # Sphere shape for leaf nodes
            marker.action = Marker.ADD
            marker.pose.position.x = node_global[0]
            marker.pose.position.y = node_global[1]
            marker.pose.position.z = 0.0  # Assuming a 2D plane
            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.1  # Size of the sphere
            marker.scale.y = 0.1
            marker.scale.z = 0.1
            marker.color.a = 1.0  # Alpha value
            marker.color.r = 1.0  # Color red for leaf nodes
            marker.color.g = 0.0
            marker.color.b = 0.988

            marker_array.markers.append(marker)

        # Publish the marker array
        self.leaf_nodes_publisher.publish(marker_array)

    def publish_tree(self):
        if not self.visualize:
            return
        marker_array = MarkerArray()

        for idx, node in enumerate(self.tree):
            node_parent_idx = self.rrt_star.parents[idx]
            if node_parent_idx == -2:
                continue  # Skip the root node, as it has no parent
            
            marker = Marker()
            marker.header.frame_id = "map"  # or your specific TF frame
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "rrt_tree"
            marker.id = idx
            marker.type = Marker.LINE_STRIP
            marker.action = Marker.ADD
            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.02  # Line width
            marker.color.a = 1.0  # Don't forget to set the alpha!
            marker.color.r = 0.0
            marker.color.g = 0.0  # Green color for the tree
            marker.color.b = 1.0

            # Start point (current node)
            start = Point()
            node_global = self.occupancy_grid.omap_to_global(node, self.map_to_car_rotation, self.map_to_car_translation)
            start.x = node_global[0]
            start.y = node_global[1]
            start.z = 0.0  # Assuming a 2D plane

            # End point (parent node)
            end = Point()
            node_parent = self.rrt_star.pts[node_parent_idx]
            node_parent_global = self.occupancy_grid.omap_to_global(node_parent, self.map_to_car_rotation, self.map_to_car_translation)
            end.x = node_parent_global[0]
            end.y = node_parent_global[1]
            end.z = 0.0  # Assuming a 2D plane

            # Assign points to the marker
            marker.points.append(start)
            marker.points.append(end)

            # Add the current line (marker) to the array
            marker_array.markers.append(marker)

        # Publish the entire tree
        self.tree_publisher.publish(marker_array)

    def publish_path(self):
        if not self.visualize:
            return
        # Create a Marker message
        marker = Marker()
        marker.header.frame_id = "map"  # Adjust according to your TF frames
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "rrt_path"
        marker.id = 0
        marker.type = Marker.LINE_STRIP  # LINE_STRIP for continuous line
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0  # Neutral orientation
        marker.scale.x = 0.05  # Width of the line
        marker.color.a = 1.0  # Alpha channel
        marker.color.r = 1.0  # Red color
        marker.color.g = 0.0
        marker.color.b = 0.0

        # Fill in the points of the path
        marker.points = []
        for node in self.path:
            # change node to global frame
            node_global = self.occupancy_grid.omap_to_global(node, self.map_to_car_rotation, self.map_to_car_translation)
            # node_car = [-(node.x - self.occupancy_grid.origin[0]) * self.occupancy_grid.resolution[0], 
            #            (node.y - self.occupancy_grid.origin[1]) * self.occupancy_grid.resolution[1]]

            # node_global = np.array([node_car[1], node_car[0], 0]) @ self.map_to_car_rotation.as_matrix().T + self.map_to_car_translation
            
            p = Point()
            p.x = node_global[0]
            p.y = node_global[1]
            p.z = 0.0  # Assuming a 2D plane, set z to 0
            marker.points.append(p)

        # Publish the Marker
        self.path_publisher.publish(marker)

    def publish_waypoints(self):
        if len(self.waypoints) == 0 or not self.visualize:
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

    def publish_goalpoint(self, goalpoint):
        if not self.visualize:
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

    def publish_rrtpoint(self, rrtpoint):
        if not self.visualize:
            return
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.id = 0
        marker.type = marker.SPHERE
        marker.action = marker.ADD
        marker.pose.position.x = rrtpoint[0]
        marker.pose.position.y = rrtpoint[1]
        marker.pose.position.z = 0.0
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.3
        marker.scale.y = 0.3
        marker.scale.z = 0.3
        marker.color.a = 1.0
        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 1.0
        self.rrtpoint_publisher.publish(marker)

    def publish_occupancy_grid(self, ps_wf: np.ndarray):
        """
        ps_wf: (N, 3)
        """
        if not self.visualize:
            return
        marker_array = MarkerArray()

        for idx in range(ps_wf.shape[0]):
            marker = Marker()
            marker.header.frame_id = "map"  # Adjust based on your TF frames
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "occupancy_grid"
            marker.id = idx
            marker.type = Marker.SPHERE  # Sphere shape for leaf nodes
            marker.action = Marker.ADD
            marker.pose.position.x = ps_wf[idx, 0]
            marker.pose.position.y = ps_wf[idx, 1]
            marker.pose.position.z = 0.0  # Assuming a 2D plane
            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.05  # Size of the sphere
            marker.scale.y = 0.05
            marker.scale.z = 0.05
            marker.color.a = 1.0  # Alpha value
            marker.color.r = 1.0  # Color red for leaf nodes
            marker.color.g = 0.0
            marker.color.b = 0.988

            marker_array.markers.append(marker)

        # Publish the marker array
        self.occupancy_grid_publisher.publish(marker_array)

def main(args=None):
    rclpy.init(args=args)
    print("RRT Initialized")
    lmpc_node = LMPC()
    rclpy.spin(lmpc_node)

    lmpc_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()