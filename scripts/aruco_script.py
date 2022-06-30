#!/usr/bin/env python3

import rclpy
from markerdetection import MarkerDetection
import sys
sys.path.insert(0,'/home/software/skyrats_ws2/src/mavbase2')
#print(sys.path)
from MAV2 import MAV2
import numpy as np

TOL = 0.4

def run():

    rclpy.init()
    rclpy.create_node("ArucoDetection")
    rclpy.get_global_executor()
    mav = MAV2()
    detection = MarkerDetection()
    cam_id = 4
    actual_x = mav.drone_pose.pose.position.x
    actual_y = mav.drone_pose.pose.position.y
    actual_z = mav.drone_pose.pose.position.z

    # aruco3 = (0, 0, 0) | aruco5 = (5, 5, 0) | aruco9 = (7, 0, 0) | aruco10 = (0, 7, 0) | aruco11 = (-3, 7, 0)
    pose_vec = [[0, 0, 0], [5, 5, 0], [7, 0, 0], [0, 7, 0], [-3, 7, 0]]
    mav.takeoff(3)
    rclpy.spin_once(mav)
    for tag_pose in range(len(pose_vec)):
        goal_x = pose_vec[tag_pose][0]
        goal_y = pose_vec[tag_pose][1]

        dist = np.sqrt((goal_x - actual_x)**2 + (goal_y - actual_y)**2)
        mav.get_logger().info("Moving to tag position...")
        mav.go_to_local(goal_x, goal_y, 3)
        mav.get_logger().info("On position")
        detection.aruco_detection(mav.cam)
        rclpy.spin_once(mav)

if __name__ == "__main__":
    run()


