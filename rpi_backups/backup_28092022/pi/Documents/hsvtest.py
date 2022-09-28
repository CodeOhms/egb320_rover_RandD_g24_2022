import numpy as np
import cv2 as cv

cap = cv.VideoCapture(0)

lower_g = np.array([8,9,80])
upper_g = np.array([25,255,255])

while True:
    ret, frame = cap.read()
    width = int(cap.get(3))
    height = int(cap.get(4))
    
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    
    mask = cv.inRange(frame, lower_g, upper_g)
    
    contours, heirachy = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
    if len(contours) != 0:
        for contour in contours:
            if cv.contourArea(contour) > 500:
                # draw rectangle
                x, y, w, h = cv.boundingRect(contour)
                cv.rectangle(frame, (x,y), (x + w, y + h), (0,0,255), 3)
    

    
    cv.imshow('frame', mask)
    cv.imshow('frame
    
    if cv.waitKey(1) == ord('q'):
        break
    
    
cap.release()
cv.destroyAllWindows()