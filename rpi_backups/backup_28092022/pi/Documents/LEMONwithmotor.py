import cv2 as cv
import numpy as np
import RPi.GPIO as GPIO
from time import sleep

#SetMode
GPIO.setmode(GPIO.BCM)

#Left Motor Setup
PWM1,IN1,IN2 = 100,4,3
GPIO.setup(IN1,GPIO.OUT) #Forwards
GPIO.setup(IN2,GPIO.OUT) #Reverse 
PWMA = GPIO.PWM(IN1,PWM1)
PWMAR = GPIO.PWM(IN2,PWM1)
PWMA.start(0)
PWMAR.start(0)

#Right Motor Setup
PWM2,IN3,IN4 = 100,17,27
GPIO.setup(IN4,GPIO.OUT) #Forwards
GPIO.setup(IN3,GPIO.OUT) #Reverse
PWMB = GPIO.PWM(IN4,PWM2)
PWMBR = GPIO.PWM(IN3,PWM2)
PWMB.start(0)
PWMBR.start(0)

#servo setup
GPIO.setup(14, GPIO.OUT)
servo = GPIO.PWM(14,50)
servo.start(0)

#setup hsv sample
lower_g = np.array([0,139,97])
upper_g = np.array([25,219,247])

#setup hsv lander
lower_l = np.array([17,139,97])
upper_l = np.array([46,219,247])

video = cv.VideoCapture(0)
search = True
ball = False

while True:
    success, img = video.read()
    image = cv.cvtColor(img, cv.COLOR_BGR2HSV)
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
                
                setspeed = 40
                
                if x_pos > 380:
                    if x_pos > 190:
                        PWMAR.ChangeDutyCycle(setspeed * 0)
                        PWMA.ChangeDutyCycle(setspeed * 0)
                        PWMB.ChangeDutyCycle(setspeed * 0.8)
                        print("te")
                    else:
                        PWMAR.ChangeDutyCycle(setspeed * 0)
                        PWMA.ChangeDutyCycle(setspeed * 0.5)
                        PWMB.ChangeDutyCycle(setspeed * 0.8)
                        print("right")
                elif x_pos < 300:
                    if x_pos < 450:
                        PWMAR.ChangeDutyCycle(setspeed * 0)
                        PWMA.ChangeDutyCycle(setspeed * 0.8)
                        PWMB.ChangeDutyCycle(setspeed * 0)
                        print("fds")
                    else:
                        PWMAR.ChangeDutyCycle(setspeed * 0)
                        PWMA.ChangeDutyCycle(setspeed * 0.8)
                        PWMB.ChangeDutyCycle(setspeed * 0.5)
                        print("left")
                else:
                    ball = True
                    if w > 300:
                        print("fire away")
                        PWMA.ChangeDutyCycle(setspeed*0.5)
                        PWMB.ChangeDutyCycle(setspeed*0.5)
                        sleep(0.8)
                        servo.ChangeDutyCycle(7)
                        sleep(0.8)
                        search = False
                        
                        print("none")
                    else:
                        PWMA.ChangeDutyCycle(setspeed * 0.5)
                        PWMB.ChangeDutyCycle(setspeed * 0.5)
                        
                
                print("x: " + str(x_pos) + ", y: " + str(y_pos) + " " + str(w))
    elif ball == False:
        PWMAR.ChangeDutyCycle(40 * 0.8)
        PWMB.ChangeDutyCycle(40 * 0.8)
        
            
    cv.imshow("webcam", img)
    #cv.imshow("mask", mask)
    
    cv.waitKey(1)