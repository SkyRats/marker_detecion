import imutils				
import numpy as np
import cv2
from imutils.perspective import four_point_transform


def calibration(img, i):

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
        
        result = cv2.dilate(imgMask, dilate_kernel)
        result = cv2.erode(result, erode_kernel)

        cv2.imshow('shapes', result)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cv2.destroyAllWindows()
    parameters = [[h, s, v], [H, S, V], [blur_size, erode_size, dilate_size]]
    print(parameters)
    return parameters


def aply_filters(img, p, otsu):

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
    cv2.imshow('blur', img)
    cv2.waitKey(0)

    # Transforma em hsv
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    cv2.imshow('hsv', hsv)
    cv2.waitKey(0)

    # Aplica filtro de cores
    mask = cv2.inRange(hsv, lower_color, upper_color)
    img_mask = cv2.bitwise_and(img, img, mask=mask)
    cv2.imshow('mask', img_mask)
    cv2.waitKey(0)

    # Erode e dilate
    erode_kernel = np.ones((erode, erode), np.float32)
    dilate_kernel = np.ones((dilate, dilate), np.float32)
    dilate = cv2.dilate(img_mask, dilate_kernel)
    erode = cv2.erode(dilate, erode_kernel)

    # Otsu thresholding
    gray = cv2.cvtColor(erode, cv2.COLOR_BGR2GRAY)
    ret, threshold = cv2.threshold(gray, otsu, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cv2.imshow('otsu', threshold)
    cv2.waitKey(0)

    return threshold


def find_potentials(image):
    contours, ret = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    shapes = []
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
        
        if len(approx) == 4 and cv2.arcLength(contour,True) > 100:
            shapes.append(approx)
    
    return shapes


def cluster(shapes):
    cluster = []
    LIM = 5
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


# Para cada quadrado -> ajustar perspectiva -> verificar se é uma base
def verify(shape, image):

    square = four_point_transform(image, shape.reshape(4, 2))
    cv2.imshow('warp perspective', square)
    cv2.waitKey(0)
    cv2.imwrite('scripts/CBRbase/zoom.png', square)
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


img = cv2.imread('CBRbase/example.jpeg')
img = imutils.resize(img, width=900)

cv2.imshow('frame', img)
cv2.waitKey(0)

initial = [[94, 70, 0], [255, 255, 255], [0, 0, 0]]

parameters = calibration(img, initial)

img_filter = aply_filters(img, parameters, 120)

list_of_potentials = find_potentials(img_filter)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, threshold = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

soma = 0
for potential in list_of_potentials:

    if verify(potential, threshold):
        M = cv2.moments(potential)
        if M['m00'] != 0.0:
            x = int(M['m10']/M['m00'])
            y = int(M['m01']/M['m00'])
            cv2.circle(img, (x, y), 4, (0, 0, 255), -1)

        soma += 1

print(soma)
cv2.imshow('final result', img)
cv2.waitKey(0)
