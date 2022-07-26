#!/usr/bin/env python3
import time
import rclpy
from markerdetection import MarkerDetection
import sys
sys.path.insert(0,'/home/alessandro/skyrats_ws2/src/mavbase2')
from MAV2 import MAV2


# Utilize a função centralize_on_aruco fornecendo o mav e as coordenadas do aruco
# Ao fornecer também um ID, o drone centraliza apenas no aruco desejado
# Isso pode ser útil no caso de uma varredura

rclpy.init()
rclpy.create_node("ArucoDetection")
rclpy.get_global_executor()
mav = MAV2()
detection = MarkerDetection()
detection.camera_topic = "/camera/image_raw"

pose_vec = [(1, 1, 0), (5.2, 4.8, 0), (7.2, 0.3, 0), (1, 6, 0), (-2.8, 7.2, 0)]
rclpy.spin_once(mav)

for tag in pose_vec:
    detection.centralize_on_aruco(mav, tag)
    time.sleep(5)



 