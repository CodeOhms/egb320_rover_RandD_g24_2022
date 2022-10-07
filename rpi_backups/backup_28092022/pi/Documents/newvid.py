import cv2 as cv
import numpy as np

lower_g = np.array([18,9,80])
upper_g = np.array([25,255,255])

video = cv.VideoCapture(0)

while True:
    success, img = video.read()
    image = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    mask = cv.inRange(image, lower_g, upper_g)
    
    contours, heirachy = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
    if len(contours) != 0:
        for contour in contours:
            if cv.contourArea(contour) > 500:
                # draw rectangle
                x, y, w, h = cv.boundingRect(contour)
                cv.rectangle(img, (x,y), (x + w, y + h), (0,0,255), 3)
                
    cv.imshow("webcam", img)
    
    cv.waitKey(1)