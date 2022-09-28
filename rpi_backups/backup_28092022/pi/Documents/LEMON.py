import cv2 as cv
import numpy as np

lower_g = np.array([0,59,160])
upper_g = np.array([180,255,255])

video = cv.VideoCapture(0) #

while True:
    success, img = video.read()
    image = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    mask = cv.inRange(image, lower_g, upper_g)
    
    contours, heirachy = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
    if len(contours) != 0:
        for contour in contours:
            if cv.contourArea(contour) > 700:
                # find bounds
                x, y, w, h = cv.boundingRect(contour)
                x_pos = (x+x+w)//2
                y_pos = (y+y+h)//2
                cv.rectangle(img, (x,y), (x + w, y + h), (0, 0, 255), 3)
                #cv.circle(img, (x_pos,y_pos) ,20,(0,0,255), 3)
                
                print("x: " + str(x_pos) + ", " + "y: " + str(y_pos))
                
    cv.imshow("webcam", img)
    cv.imshow("mask", mask)
    
    cv.waitKey(1)