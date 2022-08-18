#!/usr/bin/env python3
from sensor_msgs.msg import Image
from rclpy.node import Node
from rclpy import qos
import sys
from cv_bridge import CvBridge
import rclpy
import cv2

from markerdetection import MarkerDetection


'''
Script for testing aruco detection on real drone camera.

First, run on terminal: 
    ros2 launch usb_cam demo_launch.py

This will start the image_raw ros2 topic with camera images.
Then, this script will be able to get the messages with the subscriber and search for arucos.

To change the camera configurations, go to:
    ~/skyrats_ws2/src/usb_cam2/config

'''

ODROID = False

class camera(Node):

    def __init__(self):
        super().__init__('camera')

        self.cam = Image()
        self.bridge_object = CvBridge()
        self.camera_topic = "image_raw"
        self.cam_sub = self.create_subscription(Image, self.camera_topic, self.cam_callback, qos.qos_profile_sensor_data)
        rclpy.spin_once(self)

    def cam_callback(self, cam_data):
        self.cam = self.bridge_object.imgmsg_to_cv2(cam_data,"bgr8")
    

if __name__ == '__main__':
    rclpy.init(args=sys.argv)
    cam = camera()
    detection = MarkerDetection()

    while True:
        rclpy.spin_once(cam)
        frame = cam.cam
        list_of_arucos = detection.aruco_detector(frame)

        print(list_of_arucos)

        # Show points on frame
        if not ODROID:
            for tag in list_of_arucos:
                cX = tag[1][0]
                cY = tag[1][1]
                id = tag[0]
                cv2.circle(frame, (cX, cY), 4, (0, 0, 255), -1)
                cv2.putText(frame, str(id),(cX + 10, cY + 10),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow('image', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
