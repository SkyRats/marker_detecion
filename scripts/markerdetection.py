import numpy as np
from pyzbar.pyzbar import decode
import cv2

import argparse
import sys

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
        self.qr_debug = True
        self.frame = None
        self.det_number = 0
        self.gen_aruco = None
        self.aruco_dic = cv2.aruco.DICT_5X5_250
        self.aruco_id = None

    def qrdetection(self, vid):
        ret, self.frame = vid.read()
        while self.detection and self.det_number<=10:
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
                print("QR Code info:\n", self.qr_data)

    def qrtest(self):

        webcam = cv2.VideoCapture(0)
        self.detection = True
        self.qrdetection(webcam)
        print("QR Code info:\n", self.qr_data)
        webcam.release()
        self.qr_debug = True
        if self.qr_debug:
            text = str(self.qr_data)
            cv2.putText(self.frame, text, (self.qr_x, self.qr_y - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
            # Display the resulting frame
            cv2.imshow('frame', self.frame)


    def aruco_generator(self, id):
        
        self.aruco_id = id
        # Load the predefined dictionary
        dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_5X5_250)

        # Generate the marker
        markerImage = np.zeros((200, 200), dtype=np.uint8)
        markerImage = cv2.aruco.drawMarker(dictionary, self.aruco_id, 800, markerImage, 2)
        cv2.imwrite("marker2.png", markerImage)
        self.gen_aruco = "marker2.png"


    def aruco_detection(self):

        self.frame = cv2.imread("/home/soph/skyrats_ws/src/marker_detecion/scripts/aruco2.png")
        #Load the dictionary that was used to generate the markers.
        dictionary = cv2.aruco.Dictionary_get(self.aruco_dic)
        # Initialize the detector parameters using default values
        parameters =  cv2.aruco.DetectorParameters_create()
        # Detect the markers in the image
        markerCorners, markerIds, rejectedCandidates = cv2.aruco.detectMarkers(self.frame, dictionary, parameters=parameters)
        print("Marker corners\n", markerCorners)
        print("MarkerIds\n", markerIds)


if __name__ == "__main__":
    detectiontest = MarkerDetection()
    #detectiontest.qrtest()
    detectiontest.aruco_generator(2)
    
    detectiontest.aruco_detection()
