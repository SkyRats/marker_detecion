from imutils.video import VideoStream	
import cv2
import time			
import imutils
import numpy as np

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


def axis_test():

	aruco_dic = cv2.aruco.DICT_5X5_1000
	arucoDict = cv2.aruco.Dictionary_get(aruco_dic)
	arucoParams = cv2.aruco.DetectorParameters_create()

	calibration_matrix_path = "matrix_path"
	distortion_coefficients_path = "distortion_path"
	matrix_coefficients = np.load(calibration_matrix_path)
	distortion_coefficients = np.load(distortion_coefficients_path)

	print("[INFO] starting video stream...")
	vs = VideoStream(src=0).start()
	time.sleep(2.0)

	while True:
		img = vs.read()
		frame = imutils.resize(img, 600)

		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

		corners, ids, rejected_img_points = cv2.aruco.detectMarkers(gray, arucoDict,parameters=arucoParams,
        cameraMatrix=matrix_coefficients,
        distCoeff=distortion_coefficients)

        # If markers are detected
		if len(corners) > 0:
			for i in range(0, len(ids)):
				# Estimate pose of each marker and return the values rvec and tvec---(different from those of camera coefficients)
				rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(corners[i], 0.02, matrix_coefficients, distortion_coefficients)
				# Draw a square around the markers
				cv2.aruco.drawDetectedMarkers(frame, corners) 

				# Draw Axis
				cv2.aruco.drawAxis(frame, matrix_coefficients, distortion_coefficients, rvec, tvec, 0.01)


		# show the output frame
		cv2.imshow("Frame", frame)
		key = cv2.waitKey(1) & 0xFF

		# if the `q` key was pressed, break from the loop
		if key == ord("q"):
			break

	# cleanup
	cv2.destroyAllWindows()
	vs.stop()


def teste2():
	print("[INFO] starting video stream...")
	vs = VideoStream(src=0).start()
	
	time.sleep(2.0)
	while True:
		frame = vs.read()
		img = imutils.resize(frame, 600)
		# show the output frame
		cv2.imshow("Frame", img)
		key = cv2.waitKey(1) & 0xFF

		# if the `q` key was pressed, break from the loop
		if key == ord("q"):
			break


# camera_test()
# teste2()
axis_test()
