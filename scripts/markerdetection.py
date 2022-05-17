import numpy as np
from pyzbar.pyzbar import decode
import cv2


class MarkerDetection():
    
    def __init__(self):

        self.detection = False
        self.qr_data = ""
        self.qr_debug = True
        self.frame = None
        self.det_number = 0

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


if __name__ == "__main__":
    detectiontest = MarkerDetection()
    detectiontest.qrtest()
