import cv2 as cv
import numpy as np
import RPi.GPIO as GPIO
import time

#SetMode
GPIO.setmode(GPIO.BCM)

#Left Motor Setup
PWM1,IN1,IN2 = 100,14,15
GPIO.setup(IN1,GPIO.OUT) #Forwards
GPIO.setup(IN2,GPIO.OUT) #Reverse 
PWMA = GPIO.PWM(IN1,PWM1)
PWMAR = GPIO.PWM(IN2,PWM1)
PWMA.start(0)
PWMAR.start(0)

#Right Motor Setup
PWM2,IN3,IN4 = 100,17,18
GPIO.setup(IN4,GPIO.OUT) #Forwards
GPIO.setup(IN3,GPIO.OUT) #Reverse
PWMB = GPIO.PWM(IN4,PWM2)
PWMBR = GPIO.PWM(IN3,PWM2)
PWMB.start(0)
PWMBR.start(0)

#servo setup
GPIO.setup(4, GPIO.OUT)
servo = GPIO.PWM(4,50)
servo.start(0)

#setup hsv sample
lower_g = np.array([0,139,97])
upper_g = np.array([25,219,247])

#setup hsv lander
lower_l = np.array([17,139,97])
upper_l = np.array([46,219,247])

video = cv.VideoCapture(0)

while True:
    success, img = video.read()
    image = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    search = True
    if search == True:
        mask = cv.inRange(image, lower_g, upper_g)
    else:
        mask = cv.inRange(image, lower_l, upper_l)
    
    contours, heirachy = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
    if len(contours) != 0:
        for contour in contours:
            if cv.contourArea(contour) > 700:
                # find bounds
                x, y, w, h = cv.boundingRect(contour)
                x_pos = (x+x+w)//2
                y_pos = (y+y+h)//2
                cv.rectangle(img, (x,y), (x + w, y + h), (0, 0, 255), 3)
                cv.circle(img, (x_pos,y_pos) ,10,(0,255,0), 3)
                setspeed = 100
                
                #max 600 min 0
                if x_pos > 300:
                    setspeed = int(100*((x_pos-300)/300))
                    print(setspeed)
                else:
                    setspeed = int(100/((x_pos+300)/300))
                    print(setspeed)
                
                print("x: " + str(x_pos)) 
                
    cv.imshow("webcam", img)
    #cv.imshow("mask", mask)
    
    cv.waitKey(1)