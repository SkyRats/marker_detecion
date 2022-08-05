from imutils.video import VideoStream	
import cv2
import time			

from markerdetection import MarkerDetection


def camera_test():
	detection = MarkerDetection()

	print("[INFO] starting video stream...")
	vs = VideoStream(src=0).start()
	time.sleep(2.0)

	while True:
		frame = vs.read()
		list_arucos = detection.aruco_detector(frame)

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

camera_test()