import cv2		
import numpy as np
import pickle

def axis_test():

	aruco_dic = cv2.aruco.DICT_5X5_1000
	arucoDict = cv2.aruco.Dictionary_get(aruco_dic)
	arucoParams = cv2.aruco.DetectorParameters_create()

	f = open('./CameraCalibration.pckl', 'rb')
	(cameraMatrix, distCoeffs, _, _) = pickle.load(f)

	print("[INFO] starting video stream...")
	cam = cv2.VideoCapture(2)
	cam.set(3, 640)
	cam.set(4, 480)

	while True:
		success, frame = cam.read()

		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		corners, ids, rejected_img_points = cv2.aruco.detectMarkers(gray, arucoDict,parameters=arucoParams,
        cameraMatrix=cameraMatrix,
        distCoeff=distCoeffs)
        # If markers are detected
		if len(corners) > 0:
			for i in range(0, len(ids)):
				# Estimate pose of each marker and return the values rvec and tvec---(different from those of camera coefficients)
				rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(corners[i], 0.2, cameraMatrix, distCoeffs)
				# Draw a square around the markers
				cv2.aruco.drawDetectedMarkers(frame, corners)
				# Draw Axis
				cv2.aruco.drawAxis(frame, cameraMatrix, distCoeffs, rvec, tvec, 0.1)
				print(tvec)
				print("x: " + str(-tvec[0][0][1]) + "; y: " + str(-tvec[0][0][0]) + "; z: " + str(-tvec[0][0][2]))

		# show the output frame
		cv2.imshow("Frame", frame)
		key = cv2.waitKey(1) & 0xFF

		# if the `q` key was pressed, break from the loop
		if key == ord("q"):
			break

	# cleanup
	cv2.destroyAllWindows()
	cam.stop()

if __name__ == "__main__":
	axis_test()
