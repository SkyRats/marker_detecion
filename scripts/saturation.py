import cv2

def nothing(x):
    pass

capture = cv2.VideoCapture(0)

cv2.namedWindow('Parâmetros')
cv2.createTrackbar('Weight', 'Parâmetros', -200, 200, nothing)

while True: 
    success, frame = capture.read()
    if success == False:
        raise ConnectionError
    camera = frame.copy()
    gray = cv2.cvtColor(camera, cv2.COLOR_BGR2GRAY)
    camera_structure_fix = cv2.cvtColor(gray,cv2.COLOR_GRAY2RGB)
    
    gray_inv = cv2.bitwise_not(camera_structure_fix)
    weight = cv2.getTrackbarPos('Weight', 'Parâmetros')

    final = cv2.addWeighted(camera, weight / 100, gray_inv, 1 - weight / 100 , 0)
    cv2.imshow('Antes', camera)
    cv2.imshow('Depois', final)

    cv2.waitKey(2)
