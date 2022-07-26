from imutils.video import VideoStream	
import imutils				
import numpy as np
from pyzbar.pyzbar import decode
import cv2
import argparse
import sys
import time			
import rclpy

ARUCO_DICT = {
	"DICT_5X5_50": cv2.aruco.DICT_5X5_50,
	"DICT_5X5_100": cv2.aruco.DICT_5X5_100,
	"DICT_5X5_250": cv2.aruco.DICT_5X5_250,
	"DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
}

class MarkerDetection():
    
    def __init__(self):

        self.detection = False
        self.qr_data = ""
        self.qr_debug = False
        self.frame = None
        self.det_number = 0
        self.gen_aruco = None
        self.aruco_dic = cv2.aruco.DICT_5X5_1000
        self.arucoDict = cv2.aruco.Dictionary_get(self.aruco_dic)
        self.arucoParams = cv2.aruco.DetectorParameters_create()
        self.aruco_id = None
        self.aruco_debug = False
        self.qr_x = 0
        self.qr_y = 0
        self.qr_w = 0
        self.qr_h = 0
        self.TOL = 0.0140
        self.PID = 1/2000
        self.TARGET = (500, 400)

    def qrdetection(self, vid):
        ret, self.frame = vid.read()
        while self.detection: # and self.det_number<=10:
            if self.qr_data != "":
                self.det_number += 1
            ret, self.frame = vid.read()
            qr_result = decode(self.frame)

            if len(qr_result)>0:
                print("QR Code being detected")

                for barcode in qr_result:

                    (self.qr_x, self.qr_y, self.qr_w, self.qr_h) = barcode.rect
                    cv2.rectangle(self.frame, (self.qr_x, self.qr_y), (self.qr_x + self.qr_w, self.qr_y + self.qr_h), (0, 0, 255), 2)
                    self.qr_data = barcode.data.decode("utf-8")
                    self.qr_type = barcode.type
                    print("QR Code info: ", self.qr_data)
                    cv2.putText(self.frame, str(self.qr_data), (self.qr_x, self.qr_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
            
            if self.qr_debug:
                cv2.imshow("Frame", self.frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        # cleanup
        cv2.destroyAllWindows()

    def qrtest(self, cam, cam_id=None):
        cam_id = 0
        #webcam = cv2.VideoCapture(cam_id)
        camera = cv2.VideoCapture(cam)
        self.detection = True
        self.qr_debug = True
        self.qrdetection(camera)
        camera.release()        


    def aruco_generator(self, id):
        
        self.aruco_id = id
        aruco_size = 800
        border_size = int(aruco_size/15)
        # Load the predefined dictionary
        dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_5X5_250)

        # Generate the marker
        markerImage = np.zeros((200, 200), dtype=np.uint8)
        markerImage = cv2.aruco.drawMarker(dictionary, self.aruco_id, aruco_size, markerImage, 1)
        self.gen_aruco = cv2.copyMakeBorder(markerImage, border_size, border_size, border_size, border_size, cv2.BORDER_CONSTANT, value=[255, 255, 255])
        cv2.imwrite("markerID="+str(id)+".png", self.gen_aruco)
    

    def detect_arucos(self, frame, id=None):

        frame = imutils.resize(frame, width=1000)
        (corners, ids, rejected) = cv2.aruco.detectMarkers(frame, self.arucoDict, parameters=self.arucoParams)
        lista_arucos = []

        if len(corners) > 0:
            
            ids = ids.flatten()

            for (markerCorner, markerID) in zip(corners, ids):

                corners = markerCorner.reshape((4, 2))
                (topLeft, topRight, bottomRight, bottomLeft) = corners

                tR = (int(topRight[0]), int(topRight[1]))
                bR = (int(bottomRight[0]), int(bottomRight[1]))
                bL = (int(bottomLeft[0]), int(bottomLeft[1]))
                tL = (int(topLeft[0]), int(topLeft[1]))

                area = ( (tR[0]*bR[1] - bR[0]*tR[1]) + (bR[0]*bL[1] - bL[0]*bR[1]) + (bL[0]*tL[1] - tL[0]*bL[1]) + (tL[0]*tR[0] - tR[0]*tL[1]) ) / 2

                cX = int((tR[0] + bR[0]) / 2.0)
                cY = int((tL[1] + bR[1]) / 2.0)

                if id == None:
                    lista_arucos.append([markerID, (cX, cY), area])
                else:
                    if markerID == id:
                        lista_arucos.append([markerID, (cX, cY), area])

        return lista_arucos


    def centralize_on_aruco(self, drone, tag, aruco_id=None):

        goal_x = tag[0]
        goal_y = tag[1]
        goal_z = tag[2] + 2.0

        drone.get_logger().info("Moving to aruco region...")
        drone.go_to_local(goal_x, goal_y, goal_z)
        drone.get_logger().info("On position, waiting for stabilization...")
        
        time.sleep(5)

        drone.get_logger().info("Drone stable, starting centralization..")
        is_centralize = False

        while not is_centralize:
            
            aruco_detected = False
            timer = 0
            no_detection = 0
            while not aruco_detected:
                frame = drone.cam
                list_of_arucos = self.detect_arucos(frame, aruco_id)

                if len(list_of_arucos) > 0:
                    aruco_detected = True
                    timer = 0

                if timer > 500:
                    drone.set_vel(0, 0, 0.1)
                    print("No visible Arucos, going up...")
                    timer = 0
                    no_detection += 1

                if no_detection > 100:
                    print("Aruco not found! Going back to starting position...")
                    drone.go_to_local(0, 0, goal_z)
                    return

                timer += 1
                rclpy.spin_once(drone)

            aruco = list_of_arucos[0]
            markerID = aruco[0]

            delta_x = self.TARGET[0] - aruco[1][0]
            delta_y = self.TARGET[1] - aruco[1][1]
            area = aruco[2]

            vel_x = delta_y * self.PID
            vel_y = delta_x * self.PID
            vel_z = -(85000 - area) / 500000

            if abs(vel_x) < self.TOL:
                vel_x = 0.0
            if abs(vel_y) < self.TOL:
                vel_y = 0.0
            if abs(vel_z) < self.TOL:
                vel_z = 0.0
                    
            drone.set_vel(vel_x, vel_y, vel_z)
            print(f"Set_vel -> x: {vel_x} y: {vel_y} z: {vel_z}")

            if ((delta_x)**2 + (delta_y)**2)**0.5 < 30:
                drone.set_vel(0, 0, 0)
                is_centralize = True
                print(f"Centralizou! x: {delta_x} y: {delta_y}")
                
            rclpy.spin_once(drone)
        return markerID


if __name__ == "__main__":

    detection = MarkerDetection()
    # detectiontest.qrtest(0) # recebe o id da camera a ser testada
    # detectiontest.aruco_generator(10)

    print("[INFO] starting video stream...")
    vs = VideoStream(src=0).start()
    time.sleep(2.0)

    while True:
        frame = vs.read()
        list_arucos = detection.detect_arucos(frame)
        if len(list_arucos) > 0:
            print(list_arucos)

        for tag in list_arucos:
            cX = tag[1][0]
            cY = tag[1][1]
            id = tag[0]
            cv2.circle(frame, (cX, cY), 4, (0, 0, 255), -1)
            cv2.putText(frame, str(id),
                (cX + 10, cY + 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 255, 0), 2)

        # show the output frame
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

    # cleanup
    cv2.destroyAllWindows()
    vs.stop()

    
