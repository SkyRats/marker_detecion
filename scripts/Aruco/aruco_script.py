#!/usr/bin/env python3

import rclpy
from marker_detection import MarkerDetection
drome mavbase2 import MAV2
import numpy as np
TOL = 0.4

def run():

    mav = MAV2()
    detection = MarkerDetection()
    actual_x = mav.drone_pose.pose.position.x
    actual_y = mav.drone_pose.pose.position.y
    actual_z = mav.drone_pose.pose.position.z

    # aruco3 = (0, 0, 0) | aruco5 = (5, 5, 0) | aruco9 = (7, 0, 0) | aruco10 = (0, 7, 0) | aruco11 = (-3, 7, 0)
    pose_vec = [[0, 0, 0], [5, 5, 0], [7, 0, 0], [0, 7, 0], [-3, 7, 0]]
    mav.takeoff(3)
    for tag_pose in range(len(pose_vec)):
        goal_x = pose_vec[tag_pose[0]]
        goal_y = pose_vec[tag_pose[1]]

        dist = np.sqrt((goal_x - actual_x)**2 + (goal_y - actual_y)**2)
        mav.get_logger().info("Moving to tag position...")
        while dist > TOL:
            mav.set_position(goal_x, goal_y, actual_z)
            actual_x = mav.drone_pose.pose.position.x
            actual_y = mav.drone_pose.pose.position.y
            actual_z = mav.drone_pose.pose.position.z
            dist = np.sqrt((goal_x - actual_x)**2 + (goal_y - actual_y)**2)
        mav.get_logger().info("On position")
        detection.aruco_detection()

if __name__ == "__main__":
    run()


