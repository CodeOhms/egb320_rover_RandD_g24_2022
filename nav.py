import cv2 as cv
import numpy as np
import picamera
import RPi.GPIO as GPIO
import time
from time import sleep

debug = False
demonstration = True

def empty(img):
    pass

from time import sleep

#import the sleep from time library
MotorA_F = 14
MotorA_B = 15



soil_angle = 99
soil_distance = 0

lander_angle = 99
found_ball = False;

rock_angle = 99
rock_distance = 0


MotorB_F = 18
MotorB_B = 17


class PositionalServo:

    def __init__(self, outpin = 4):
            self.pin = outpin
            GPIO.setwarnings(False)
            GPIO.setmode(GPIO.BCM) #set pin numbering system
            GPIO.setup(self.pin,GPIO.OUT)
            self.servo = GPIO.PWM(self.pin, 100)    # Set to a frequency of 50Hz
            self.servo.start(0) # Start at angle 0
            
    def change_duty(self, duty):
        self.action =  duty
        self.servo.ChangeDutyCycle(self.action)

    def motor_up(self):
        Up = 9
        self.change_duty(Up)

    def motor_down(self):
        Down = 6.75
        self.change_duty(Down)
        
    def collect(self):
        self.change_duty(Up)
        
            
    def flip_rock(self):
        self.motor_up()
        sleep(4)
        # Integrate code to drive forwards onto the rock
        self.motor_up()
        
# Integrate code for alignment and correction
move_servo = PositionalServo()
move_servo.motor_down()
#move_servo.flip_rock()

#declare the GPIO 18 pin for the output of LED
GPIO.setmode(GPIO.BCM)
GPIO.setup(MotorA_F,GPIO.OUT)
GPIO.setup(MotorA_B, GPIO.OUT)
GPIO.setup(MotorB_F,GPIO.OUT)
GPIO.setup(MotorB_B, GPIO.OUT)
GPIO.setup


#define the behaviour of the MotorA_F as output

#ignore the warnings

pwm = GPIO.PWM(MotorA_F,100)

pwm2 = GPIO.PWM(MotorA_B, 100)

pwm3 = GPIO.PWM(MotorB_F,100)

pwm4 = GPIO.PWM(MotorB_B, 100)

#create the pwm instance with frequency 1000 Hz

pwm.start(0)
pwm2.start(0)
pwm3.start(0)
pwm4.start(0)

## Still Image
img = cv.imread('Photos/Image.png')
frame = cv.imread('CV/frames/frame_000048.png')

def changeRes(width,height):
    capture.set(3,width)
    capture.set(4,height)

## Video Image
capture = cv.VideoCapture(0)
changeRes(640,480)

if debug == True:
    width_ = capture.get(cv.CAP_PROP_FRAME_WIDTH)
    height_ = capture.get(cv.CAP_PROP_FRAME_HEIGHT)
    print(width_,height_)

# Soil HSV Values
S_hue_min = 0
S_hue_max = 24
S_sat_min = 105
S_sat_max = 251
S_val_min = 64
S_val_max = 255

# Rock HSV Values
R_hue_min = 90
R_hue_max = 110
R_sat_min = 110
R_sat_max = 255
R_val_min = 26
R_val_max = 148

# Obstacle HSV Values
O_hue_min = 32  
O_hue_max = 87
O_sat_min = 99
O_sat_max = 255
O_val_min = 24
O_val_max = 154

# Lander HSV Values
L_hue_min = 23
L_hue_max = 32
L_sat_min = 44
L_sat_max = 255
L_val_min = 180
L_val_max = 255

# HSV Thresholding (Trackbar app)
if debug==True:
    cv.namedWindow('Trackbar')
    cv.resizeWindow('Trackbar',600,300)
    cv.createTrackbar('hue_min','Trackbar',0,179,empty)
    cv.createTrackbar('hue_max','Trackbar',179,179,empty)
    cv.createTrackbar('sat_min','Trackbar',0,255,empty)
    cv.createTrackbar('sat_max','Trackbar',255,255,empty)
    cv.createTrackbar('val_min','Trackbar',0,255,empty)
    cv.createTrackbar('val_max','Trackbar',255,255,empty)

while True:
    isTrue, frame = capture.read()
    start = time.time()
    
    ## Get Resolution
    height, width = frame.shape[:2]
    
    ## Erosion/Dilation Kernel Size
    kernel = np.ones((5, 5), np.uint8)

    ## BGR to HSV Colourspaces
    hsv_frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    if debug==True:
        ## Get threshold values from trackbar    
        hue_min = cv.getTrackbarPos('hue_min','Trackbar')
        hue_max = cv.getTrackbarPos('hue_max','Trackbar')
        sat_min = cv.getTrackbarPos('sat_min','Trackbar')
        sat_max = cv.getTrackbarPos('sat_max','Trackbar')
        val_min = cv.getTrackbarPos('val_min','Trackbar')
        val_max = cv.getTrackbarPos('val_max','Trackbar')

    ## Apply soil colour mask
    soilMask = cv.inRange(hsv_frame, (S_hue_min, S_sat_min, S_val_min), (S_hue_max, S_sat_max, S_val_max)) 
    soilErode = cv.erode(soilMask,kernel,iterations=1) 
    soilDilate = cv.dilate(soilErode,kernel,iterations=1) 
    # contours, heirarchies = cv.findContours(mask, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
    # contours, heirarchies = cv.findContours(mask, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
    soilContours, heirarchies = cv.findContours(soilDilate, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    for cnt in soilContours:
        area = cv.contourArea(cnt)
        if (area>50):
            peri = cv.arcLength(cnt, True)
            approx = cv.approxPolyDP(cnt, 0.02*peri,True)
            x,y,w,h = cv.boundingRect(cnt)
            # cv.rectangle(frame,(x,y),(x+w,y+h), (0,255,0),2)
            # cv.drawContours(frame,contours,-1,(0,255,0),2)
            if len(approx)>=6:
                # cv.putText(frame, 'Soil Sample '+'('+str(width-x+w/2)+','+str(y+h/2)+')',(x+w,y+h+20),cv.FONT_HERSHEY_COMPLEX,0.7,(0,255,0),1)
                # cv.putText(frame, 'Soil Sample at '+str(h),(x+w,y+h+20),cv.FONT_HERSHEY_COMPLEX,0.7,(31,95,255),1)
                # cv.fillPoly(frame, contours, (51,153,255))
                # cv.drawContours(frame,contours,-1,(0,255,0),2)
                (x,y),radius = cv.minEnclosingCircle(cnt)
                center = (int(x),int(y))
                angle = round((x-width/2)/(width/2)*30,1)
                F = (180*150)/70
                distance = round((F*(4.3/2))/radius,1)
                cv.putText(frame,'Soil Sample at: '+str(distance)+'cm, '+ str(angle)+' degrees',((int(x+w/2+5),int(y+h/2+5))),cv.FONT_HERSHEY_COMPLEX,0.7,(31,95,255),1)  
                radius = int(radius)
                cv.circle(frame,center,radius,(31,95,255),2)
                
                soil_angle = angle
                soil_distance = distance
                # cv.drawContours(frame, [approx], -1, (0,255,0))

    ## Apply rock colour mask
    rockMask = cv.inRange(hsv_frame, (R_hue_min, R_sat_min, R_val_min), (R_hue_max, R_sat_max, R_val_max))  
    rockErode = cv.erode(rockMask,kernel,iterations=1) 
    rockDilate = cv.dilate(rockErode,kernel,iterations=1) 
    # contours, heirarchies = cv.findContours(mask, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
    rockContours, heirarchies = cv.findContours(rockDilate, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    for cnt in rockContours:
        area = cv.contourArea(cnt)
        if (area>250):
            peri = cv.arcLength(cnt, True)
            approx = cv.approxPolyDP(cnt, 1*peri,True)
            x,y,w,h = cv.boundingRect(cnt)
            #print(str(x)+' '+str(y))
            # cv.rectangle(frame,(x,y),(x+w,y+h), (0,255,0),2)
            cv.drawContours(frame,cnt,-1,(255,0,0),2)
            angle = round(30.0*((x+w/2)-width/2)/(width/2),1)
            F = (180*150)/70
            distance = round((F*7.0)/h,1)
            
            rock_angle = angle
            rock_distance = distance
            cv.putText(frame,'Rock at: '+str(distance)+'cm, '+ str(angle)+' degrees',((int(x+w/2+5),int(y+h/2+5))),cv.FONT_HERSHEY_COMPLEX,0.7,(255,0,0),1)  


    ## Apply obstacle colour mask
    obstacleMask = cv.inRange(hsv_frame, (O_hue_min, O_sat_min, O_val_min), (O_hue_max, O_sat_max, O_val_max))  
    obstacleErode = cv.erode(obstacleMask,kernel,iterations=1) 
    obstacleDilate = cv.dilate(obstacleErode,kernel,iterations=1) 
    # obstacleContours, heirarchies = cv.findContours(obstacleMask, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
    obstacleContours, heirarchies = cv.findContours(obstacleDilate, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    for cnt in obstacleContours:
        area = cv.contourArea(cnt)
        x,y,w,h = cv.boundingRect(cnt)
        if (area>500):
            peri = cv.arcLength(cnt, True)
            approx = cv.approxPolyDP(cnt, 0.02*peri,True)
            x,y,w,h = cv.boundingRect(cnt)
            # cv.rectangle(frame,(x,y),(x+w,y+h), (0,255,0),2)
            cv.drawContours(frame,cnt,-1,(0,255,0),2)
            angle = round(30.0*((x+w/2)-width/2)/(width/2),1)
            F = (180*150)/70
            distance = round((F*15.5)/h,1)
            cv.putText(frame,'Obstacle at: '+str(distance)+'cm, '+ str(angle)+' degrees',((int(x+w/2+5),int(y+h/2+5))),cv.FONT_HERSHEY_COMPLEX,0.7,(0,255,0),1)  


    ## Apply lander colour mask
    landerMask = cv.inRange(hsv_frame, (L_hue_min, L_sat_min, L_val_min), (L_hue_max, L_sat_max, L_val_max))  
    landerErode = cv.erode(landerMask,kernel,iterations=1) 
    landerDilate = cv.dilate(landerErode,kernel,iterations=1) 
    # landerContours, heirarchies = cv.findContours(landerMask, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
    landerContours, heirarchies = cv.findContours(landerDilate, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    for cnt in landerContours:
        area = cv.contourArea(cnt)
        x,y,w,h = cv.boundingRect(cnt)
        if (area>300):
            peri = cv.arcLength(cnt, True)
            approx = cv.approxPolyDP(cnt, 0.02*peri,True)
            x,y,w,h = cv.boundingRect(cnt)
            # cv.rectangle(frame,(x,y),(x+w,y+h), (0,255,0),2)
            cv.drawContours(frame,cnt,-1,(0,255,255),2)
            angle = round(30.0*((x+w/2)-width/2)/(width/2),1)
            lander_angle = angle
            F = (180*150)/70
            distance = round((F*5.5)/h,1)
            #distance = round(1/height*1000,1)
            cv.putText(frame,'Lander at: '+str(distance)+'cm, '+ str(angle)+' degrees',((int(x+w/2+5),int(y+h/2+5))),cv.FONT_HERSHEY_COMPLEX,0.7,(0,255,255),1)  

    ## imshow commands
    if debug==True:
        # cv.imshow('Soil',rockMask)
        # cv.imshow('Soil Erode',rockErode)
        # cv.imshow('Soil Dilate',rockDilate)
        cv.imshow('Soil',soilDilate)
        cv.imshow('Rock',rockDilate)
        cv.imshow('Obstacle',obstacleDilate)
        cv.imshow('Lander',landerDilate)
    if demonstration==True:
        ## Get framerate
        finish = time.time()
        duration = finish-start
        #print('Duration = '+str(duration))
        #print(str(finish))
        fps = 1/duration
        #print(str(round(fps,2)))
        cv.putText(frame, str(round(fps,2))+' FPS',(20,20),cv.FONT_HERSHEY_COMPLEX,0.7,(0,255,0),1)
    
        ## Unprocessed BGR capture with mask overlays
        cv.imshow('Video', frame)    
    print(str(soil_angle) + ", " + str(soil_distance))
    
    
    
    
    if soil_angle < -4:
        
        if soil_angle < -25:
            print("FKN TURN")
            pwm.ChangeDutyCycle(40)# right f
            pwm2.ChangeDutyCycle(0) # right b
            pwm3.ChangeDutyCycle(0) # left f
            pwm4.ChangeDutyCycle(0) # left b            
        elif -25 < soil_angle < -10:
            print("LEFT")
            pwm.ChangeDutyCycle(40)# right f
            pwm2.ChangeDutyCycle(0) # right b
            pwm3.ChangeDutyCycle(25) # left f
            pwm4.ChangeDutyCycle(0) # left b         
        else:
            print("left")
            pwm.ChangeDutyCycle(40) # right f
            pwm2.ChangeDutyCycle(0) # right b
            pwm3.ChangeDutyCycle(30) # left f
            pwm4.ChangeDutyCycle(0) # left b
    elif soil_angle > 4:
        if soil_angle > 25:
            print("RIGHT LASA")
            pwm.ChangeDutyCycle(0) # right f
            pwm2.ChangeDutyCycle(0) # right b
            pwm3.ChangeDutyCycle(40) # left f
            pwm4.ChangeDutyCycle(0) # left b            
        elif 25 > soil_angle > 10:
            print("RIGHT")
            pwm.ChangeDutyCycle(25) # right f
            pwm2.ChangeDutyCycle(0) # right b
            pwm3.ChangeDutyCycle(40) # left f
            pwm4.ChangeDutyCycle(0) # left b

        else:
            print("right")
            pwm.ChangeDutyCycle(30) # right f
            pwm2.ChangeDutyCycle(0) # right b
            pwm3.ChangeDutyCycle(40) # left f
            pwm4.ChangeDutyCycle(0) # left b
    else:
        if soil_distance < 7: 
            pwm.ChangeDutyCycle(0) # right f
            pwm2.ChangeDutyCycle(0) # right b
            pwm3.ChangeDutyCycle(0) # left f
            pwm4.ChangeDutyCycle(0) # left b
            print("stop bitch")
            
        else:
            pwm.ChangeDutyCycle(40) # right f
            pwm2.ChangeDutyCycle(0) # right b
            pwm3.ChangeDutyCycle(40) # left f
            pwm4.ChangeDutyCycle(0) # left b            

        
    
    if cv.waitKey(20) & 0xFF==ord('d'):
        break
capture.release()
cv.destroyAllWindows()
pwm.stop()
pwm2.stop()
pwm3.stop()
pwm4.stop()
GPIO.cleanup()

