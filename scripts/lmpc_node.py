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
from dataclasses import dataclass
from matplotlib import pyplot as plt
from copy import deepcopy
from scipy.spatial.transform import Rotation as R
import tf2_ros
from visualization_msgs.msg import Marker, MarkerArray
from rclpy.parameter import Parameter, ParameterType
from sensor_msgs.msg import PointCloud
from occupancy_grid import CarOccupancyGrid
# from rrt_star import RRT_Star

nx = 6 # dim of state space
nu = 2 # dim of control space

@dataclass
class Sample:
    x: np.ndarray
    u: np.ndarray
    s: float
    time: int
    iter: int
    cost: int

# class def for RRT
class LMPC(Node):
    def __init__(self):
        super().__init__('lmpc_node')
        



def main(args=None):
    rclpy.init(args=args)
    print("RRT Initialized")
    lmpc_node = LMPC()
    rclpy.spin(lmpc_node)

    lmpc_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()