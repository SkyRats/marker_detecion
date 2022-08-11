import cv2
from imutils.video import VideoStream
import time

from markerdetection import MarkerDetection


detection = MarkerDetection()

print("[INFO] starting video stream...")
vs = VideoStream(src=8).start()
time.sleep(2.0)

initial = [[0, 0, 0], [255, 255, 255], [0, 0, 0]]

# parameters = detection.calibration(vs, initial)
parameters = initial

while True:
	frame = vs.read()
	list_of_bases = detection.base_detection(frame, parameters)
	pixels = []
	for pixel in list_of_bases:
		pixels.append(pixel)
        	# cv2.circle(frame, pixel, 5, (0, 0, 255), -1)
	# cv2.imshow('shapes', frame)
	print(pixels)
	key = cv2.waitKey(1) & 0xFF
	if key == ord("q"):
		break
