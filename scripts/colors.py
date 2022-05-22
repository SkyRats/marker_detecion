import cv2
import numpy as np

def get_mask(hsv , lower_color , upper_color):
    lower = np.array(lower_color)
    upper = np.array(upper_color)
    
    mask = cv2.inRange(hsv , lower, upper)

    return mask

capture = cv2.VideoCapture(0)


while True: 
    success, frame = capture.read()
    if success == False:
        raise ConnectionError

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask = get_mask(hsv, [160, 100, 20], [179, 255, 255]) + get_mask(hsv, [0, 100, 20], [10, 255, 255])
    
    result = cv2.bitwise_and(frame , frame , mask= mask)

    cv2.imshow('frame', result)

    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()

cv2.waitKey(0)