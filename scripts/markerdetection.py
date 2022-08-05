import numpy as np
from pyzbar.pyzbar import decode
import cv2
import time			
import rclpy

from CBRbase.crossdetection import aply_filters, find_potentials, verify


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


    def calibration(self, cam, i):

        def nothing(x):
            pass

        cv2.namedWindow("Parâmetros")
        cv2.createTrackbar('h', 'Parâmetros',i[0][0],255,nothing)
        cv2.createTrackbar('s', 'Parâmetros',i[0][1],255,nothing)
        cv2.createTrackbar('v', 'Parâmetros',i[0][2],255,nothing)

        cv2.createTrackbar('H', 'Parâmetros',i[1][0],255,nothing)
        cv2.createTrackbar('S', 'Parâmetros',i[1][1],255,nothing)
        cv2.createTrackbar('V', 'Parâmetros',i[1][2],255,nothing)

        cv2.createTrackbar('Blur', 'Parâmetros', i[2][0], 100, nothing)
        cv2.createTrackbar('Erode', 'Parâmetros', i[2][1], 100, nothing)
        cv2.createTrackbar('Dilate', 'Parâmetros', i[2][2], 100, nothing)

        while True:

            img = cam.read()

            h = cv2.getTrackbarPos('h', 'Parâmetros')
            s = cv2.getTrackbarPos('s', 'Parâmetros')
            v = cv2.getTrackbarPos('v', 'Parâmetros')

            H = cv2.getTrackbarPos('H', 'Parâmetros')
            S = cv2.getTrackbarPos('S', 'Parâmetros')
            V = cv2.getTrackbarPos('V', 'Parâmetros')

            blur_size = cv2.getTrackbarPos('Blur', 'Parâmetros')
            erode_size = cv2.getTrackbarPos('Erode', 'Parâmetros')
            dilate_size = cv2.getTrackbarPos('Dilate', 'Parâmetros')

            blur_value = (blur_size, blur_size)
            erode_kernel = np.ones((erode_size, erode_size), np.float32)
            dilate_kernel = np.ones((dilate_size, dilate_size), np.float32)

            lower = [h, s, v]
            upper = [H, S, V]

            lower_color = np.array(lower)
            upper_color = np.array(upper)

            if blur_size != 0:
                blur = cv2.blur(img,blur_value)
            else:
                blur = img

            hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, lower_color, upper_color)

            imgMask = cv2.bitwise_and(blur, blur, mask=mask)
            
            dilate = cv2.dilate(imgMask, dilate_kernel)
            erode = cv2.erode(dilate, erode_kernel)

            cv2.imshow('color calibration', erode)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

        cv2.destroyAllWindows()
        parameters = [[h, s, v], [H, S, V], [blur_size, erode_size, dilate_size]]
        print(parameters)
        return parameters


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
    

    def aruco_detector(self, frame, id=None):
        
        # Resize frame and search for arucos
        frame = imutils.resize(frame, width=1000)
        cv2.imshow("frame", frame)
        cv2.waitKey(0) 
        cv2.destroyAllWindows() 

        (corners, ids, rejected) = cv2.aruco.detectMarkers(frame, self.arucoDict, parameters=self.arucoParams)
        lista_arucos = []

        if len(corners) > 0:
            
            ids = ids.flatten()

            for (markerCorner, markerID) in zip(corners, ids):

                corners = markerCorner.reshape((4, 2))
                (topLeft, topRight, bottomRight, bottomLeft) = corners

                # Marker corners
                tR = (int(topRight[0]), int(topRight[1]))
                bR = (int(bottomRight[0]), int(bottomRight[1]))
                bL = (int(bottomLeft[0]), int(bottomLeft[1]))
                tL = (int(topLeft[0]), int(topLeft[1]))

                rect = cv2.rectangle(frame, tL, bR, (255, 0, 0), 2)
                cv2.imshow('rect', rect)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

                # Compute marker area on camera
                area = ( (tR[0]*bR[1] - bR[0]*tR[1]) + (bR[0]*bL[1] - bL[0]*bR[1]) + (bL[0]*tL[1] - tL[0]*bL[1]) + (tL[0]*tR[0] - tR[0]*tL[1]) ) / 2

                # Find the Marker center
                cX = int((tR[0] + bL[0]) / 2.0)
                cY = int((tR[1] + bL[1]) / 2.0)

                # Append detected aruco data
                if id == None:
                    lista_arucos.append([markerID, (cX, cY), area])
                else:
                    if markerID == id:
                        lista_arucos.append([markerID, (cX, cY), area])

        return lista_arucos


    def centralize_on_aruco(self, drone, tag, dz, aruco_id=None):

        '''
        Function parameters:
        drone    -> MAV2 object
        tag      -> (x, y, z) position of aruco
        dz       -> desired drone relative height to aruco
        aruco_id -> only centralize on specific ID

        '''
        # Go to (x, y, z) aproximate coordinates of the Aruco
        goal_x = tag[0]
        goal_y = tag[1]
        goal_z = tag[2] + 2.0
        goal_z = tag[2] + dz

        drone.get_logger().info("Moving to aruco region...")
        drone.go_to_local(goal_x, goal_y, goal_z)

        # Wait for stabilization
        drone.get_logger().info("On position, waiting for stabilization...")
        time.sleep(5)
        drone.get_logger().info("Drone stable, starting centralization..")

        is_centralize = False
        while not is_centralize:
            
            # Loop over frames to search for markers
            # If no markers were found, try to search from above
            aruco_detected = False
            timer = 0
            no_detection = 0
            while not aruco_detected:
                frame = drone.cam
                list_of_arucos = self.aruco_detector(frame, aruco_id)

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

            # Calculate the PID errors
            delta_x = self.TARGET[0] - aruco[1][0]
            delta_y = self.TARGET[1] - aruco[1][1]
            delta_area = aruco[2] - 85000

            # Adjust velocity
            drone.camera_pid(delta_x, delta_y, delta_area)

            # End centralization if the marker is close enough to the camera center
            if ((delta_x)**2 + (delta_y)**2)**0.5 < 30:
                drone.set_vel(0, 0, 0)
                is_centralize = True
                print(f"Centralized! x: {delta_x} y: {delta_y}")
                
            rclpy.spin_once(drone)
        return markerID
    

    def cross_detector(self, img, p):

        lower = p[0]
        upper = p[1]

        erode = p[2][0]
        dilate = p[2][1]

        lower_color = np.array(lower)
        upper_color = np.array(upper)

        img = cv2.blur(img,(2,2))


    def base_detection(self, img, parameters):

        img_filter = aply_filters(img, parameters)

        list_of_potentials = find_potentials(img_filter)

        gray = cv2.cvtColor(erode, cv2.COLOR_BGR2GRAY)
        # gray = cv2.Canny(erode, 200, 300)
        
        # setting threshold of gray image
        res, threshold = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # using a findContours() function
        contours, res = cv2.findContours(
            threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        i = 0
        shapes = []
        # list for storing names of shapes
        result = []
        for contour in contours:
            if i == 0:
                i = 1
                continue
        
            # cv2.approxPloyDP() function to approximate the shape
            approx = cv2.approxPolyDP(
                contour, 0.01 * cv2.arcLength(contour, True), True)
            
            if len(approx) == 4 and cv2.arcLength(contour,True) > 500:
                result.append((x,y))

        for potential in list_of_potentials:

            if verify(potential, img_filter):
                M = cv2.moments(potential)
                if M['m00'] != 0.0:
                    x = int(M['m10']/M['m00'])
                    y = int(M['m01']/M['m00'])
                    shapes.append([(x, y), len(approx)])
            
        print(shapes)
        return result

        def cluster(shapes):
            cluster = []
            LIM = 30
            i = 0
            while i < len(shapes):
                coord1 = shapes[i][0]
                cluster = [coord1]
                x_tot = coord1[0]
                y_tot = coord1[1]
                x_min = coord1[0] - LIM
                x_max = coord1[0] + LIM
                y_min = coord1[1] - LIM
                y_max = coord1[1] + LIM
                for j in range(len(shapes) - (i+1)):
                    coord2 = shapes[j + (i+1)][0]
                    if x_min <= coord2[0] <= x_max and y_min <= coord2[1] <= y_max:
                        cluster.append(coord2)
                        x_tot += coord2[0]
                        y_tot += coord2[1]
                if len(cluster) >= 2:
                    return (int(x_tot/len(cluster)), int(y_tot/len(cluster)))
                i += 1
            return None

        center = cluster(shapes)
        if center != None:
            cv2.circle(gray, center, 8, (0, 0, 255), -1)
        result = imutils.resize(res, width=800)
        return result


    def calibracao(self, camera, i):
        
        def nothing(x):
            pass

        cv2.namedWindow("Parâmetros")
        cv2.createTrackbar('h', 'Parâmetros',i[0][0],255,nothing)
        cv2.createTrackbar('s', 'Parâmetros',i[0][1],255,nothing)
        cv2.createTrackbar('v', 'Parâmetros',i[0][2],255,nothing)

        cv2.createTrackbar('H', 'Parâmetros',i[1][0],255,nothing)
        cv2.createTrackbar('S', 'Parâmetros',i[1][1],255,nothing)
        cv2.createTrackbar('V', 'Parâmetros',i[1][2],255,nothing)

        cv2.createTrackbar('Erode', 'Parâmetros', i[2][0], 100, nothing)
        cv2.createTrackbar('Dilate', 'Parâmetros', i[2][1], 100, nothing)

        while True:
            
            img = camera.read()

            h = cv2.getTrackbarPos('h', 'Parâmetros')
            s = cv2.getTrackbarPos('s', 'Parâmetros')
            v = cv2.getTrackbarPos('v', 'Parâmetros')

            H = cv2.getTrackbarPos('H', 'Parâmetros')
            S = cv2.getTrackbarPos('S', 'Parâmetros')
            V = cv2.getTrackbarPos('V', 'Parâmetros')

            erode_size = cv2.getTrackbarPos('Erode', 'Parâmetros')
            dilate_size = cv2.getTrackbarPos('Dilate', 'Parâmetros')

            erode_kernel = np.ones((erode_size, erode_size), np.float32)
            dilate_kernel = np.ones((dilate_size, dilate_size), np.float32)

            lower = [h, s, v]
            upper = [H, S, V]

            lower_color = np.array(lower)
            upper_color = np.array(upper)

            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, lower_color, upper_color)

            imgMask = cv2.bitwise_and(img, img, mask=mask)
            
            result = cv2.dilate(imgMask, dilate_kernel)
            result = cv2.erode(result, erode_kernel)

            result = imutils.resize(result, width=1000)
            # displaying the image after drawing contours
            cv2.imshow('shapes', result)
            key = cv2.waitKey(1) & 0xFF

            # if the `q` key was pressed, break from the loop
            if key == ord("q"):
                break
        
        # cv2.waitKey(0)
        cv2.destroyAllWindows()
        parameters = [[h, s, v], [H, S, V], [erode_size, dilate_size]]
        print(parameters)
        return parameters


if __name__ == "__main__":
    detection = MarkerDetection()
    img = cv2.imread('/home/software/skyrats_ws2/src/marker_detection/scripts/print2.png')
    detection.aruco_detector(img)

    def aruco_test():
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
    
    
    def cross_test():
        detection = MarkerDetection()

        print("[INFO] starting video stream...")
        vs = VideoStream(src=0).start()
        time.sleep(2.0)

        teste1 = [[82, 186, 0], [119, 255, 255], [0, 4]]
        detection = MarkerDetection()
        parameters = detection.calibracao(vs, teste1)

        while True:
            cam = vs.read()
            result = detection.cross_detector(cam, parameters)

            # displaying the image after drawing contours
            cv2.imshow('shapes', result)
            key = cv2.waitKey(1) & 0xFF

            # if the `q` key was pressed, break from the loop
            if key == ord("q"):
                break
        
        # cv2.waitKey(0)
        cv2.destroyAllWindows()

    # aruco_test()
    #cross_test()
                   
