import numpy as np
from pyzbar.pyzbar import decode
import cv2
import argparse
import sys
from imutils.video import VideoStream	# Webcam
import imutils				# Resize images
import time				# Delay

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
        self.aruco_dic = cv2.aruco.DICT_5X5_250
        self.aruco_id = None
        self.aruco_debug = False
        self.qr_x = 0
        self.qr_y = 0
        self.qr_w = 0
        self.qr_h = 0

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

    def aruco_detection(self, cam):

        #self.frame = self.gen_aruco
        self.detection = True
        #vs = VideoStream(cam_id).start()
        vs = cam

        #Load the dictionary that was used to generate the markers.
        dictionary = cv2.aruco.Dictionary_get(self.aruco_dic)

        arucoParams = cv2.aruco.DetectorParameters_create()

        first_detection = True

        while self.detection and self.det_number <10: #and self.det_number <= 150:

            # grab the frame from the threaded video stream and resize it to have a maximum width of 1000 pixels
            frame = cam
            frame = imutils.resize(frame, width=1000)

            # detect ArUco markers in the input frame
            (corners, ids, rejected) = cv2.aruco.detectMarkers(frame, dictionary, parameters=arucoParams)
            
            # verify *at least* one ArUco marker was detected
            if len(corners) > 0:
                
                self.det_number += 1
                # flatten the ArUco IDs list
                ids = ids.flatten()

                # loop over the detected ArUCo corners
                for (markerCorner, markerID) in zip(corners, ids):

                    # extract the marker corners
                    corners = markerCorner.reshape((4, 2))
                    (topLeft, topRight, bottomRight, bottomLeft) = corners

                    # convert each of the (x, y)-coordinate pairs to integers
                    topRight = (int(topRight[0]), int(topRight[1]))
                    bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
                    bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
                    topLeft = (int(topLeft[0]), int(topLeft[1]))

                    # draw the bounding box of the ArUCo detection
                    cv2.line(frame, topLeft, topRight, (0, 255, 0), 2)
                    cv2.line(frame, topRight, bottomRight, (0, 255, 0), 2)
                    cv2.line(frame, bottomRight, bottomLeft, (0, 255, 0), 2)
                    cv2.line(frame, bottomLeft, topLeft, (0, 255, 0), 2)
            
                    # compute and draw the center (x, y)-coordinates of the ArUco marker
                    cX = int((topLeft[0] + bottomRight[0]) / 2.0)
                    cY = int((topLeft[1] + bottomRight[1]) / 2.0)
                    cv2.circle(frame, (cX, cY), 4, (0, 0, 255), -1)

                    # draw the ArUco marker ID on the frame
                    cv2.putText(frame, str(markerID),
                        (topLeft[0], topLeft[1] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 2)

                self.aruco_id = ids
                number_of_arucos = len(self.aruco_id)
                last_aruco_detected = self.aruco_id[number_of_arucos-1]
                if first_detection:
                    print("ArUco ID: ", self.aruco_id)
                    first_detection = False
                    old_arucos = self.aruco_id
                else:
                    for old_aruco in old_arucos:
                        for new_aruco in self.aruco_id:
                            if old_aruco != new_aruco:
                                first_detection = True

            # show the output frame
            cv2.imshow("Frame", frame)
            key = cv2.waitKey(1) & 0xFF

            # if the `q` key was pressed, break from the loop
            if key == ord("q"):
                break

        # cleanup
        cv2.destroyAllWindows()


if __name__ == "__main__":
    detectiontest = MarkerDetection()
    detectiontest.qrtest(0) # recebe o id da camera a ser testada
    #detectiontest.aruco_generator(10)
    
    #detectiontest.aruco_detection()
