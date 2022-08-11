from imutils.perspective import four_point_transform
import numpy as np
import cv2


# Apply filters with given parameters on image
def aply_filters(img, p):

    lower = p[0]
    upper = p[1]

    blur = (p[2][0], p[2][0])
    erode = p[2][1]
    dilate = p[2][2]

    lower_color = np.array(lower)
    upper_color = np.array(upper)

    # Aplica Blur
    if p[2][0] != 0:
        img = cv2.blur(img,blur)

    # Transforma em hsv
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Aplica filtro de cores
    mask = cv2.inRange(hsv, lower_color, upper_color)
    img_mask = cv2.bitwise_and(img, img, mask=mask)

    # Erode e dilate
    erode_kernel = np.ones((erode, erode), np.float32)
    dilate_kernel = np.ones((dilate, dilate), np.float32)
    dilate = cv2.dilate(img_mask, dilate_kernel)
    erode = cv2.erode(dilate, erode_kernel)

    # Otsu thresholding
    # gray = cv2.cvtColor(erode, cv2.COLOR_BGR2GRAY)
    # ret, threshold = cv2.threshold(gray, otsu, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Canny filter
    canny = cv2.Canny(erode, 200, 300)

    return canny


def find_potentials(image):
    contours, ret = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    shapes = []
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
        
        if len(approx) == 4 and cv2.arcLength(contour,True) > 200:
            shapes.append(approx)
    
    return shapes


def cluster(shapes):
    cluster = []
    LIM = 10
    i = 0
    while i < len(shapes):
        coord1 = shapes[i]
        cluster = [coord1]
        x_tot = coord1[0]
        y_tot = coord1[1]
        x_min = coord1[0] - LIM
        x_max = coord1[0] + LIM
        y_min = coord1[1] - LIM
        y_max = coord1[1] + LIM
        for j in range(len(shapes) - (i+1)):
            coord2 = shapes[j + (i+1)]
            if x_min <= coord2[0] <= x_max and y_min <= coord2[1] <= y_max:
                cluster.append(coord2)
                x_tot += coord2[0]
                y_tot += coord2[1]
        if len(cluster) >= 3:
            return True
        i += 1
    return False


def verify(shape, image):
    square = four_point_transform(image, shape.reshape(4, 2))
    contours, ret = cv2.findContours(
        square, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    shapes = []
    for contour in contours:

        approx = cv2.approxPolyDP(
            contour, 0.01 * cv2.arcLength(contour, True), True)
        
        if len(approx) < 20 and cv2.arcLength(contour,True) > 100:
            M = cv2.moments(contour)
            if M['m00'] != 0.0:
                x = int(M['m10']/M['m00'])
                y = int(M['m01']/M['m00'])
                shapes.append((x, y))

    return cluster(shapes)
