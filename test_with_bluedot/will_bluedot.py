
# Import Libraries
import RPi.GPIO as GPIO
from time import sleep
from bluedot import BlueDot
from signal import pause
import sys

bd = BlueDot()
   
#SetMode
GPIO.setmode(GPIO.BCM)

#Left Motor Setup
PWM1,IN1,IN2 = 100,27,22
GPIO.setup(IN1,GPIO.OUT) #Forwards
GPIO.setup(IN2,GPIO.OUT) #Reverse
PWMA = GPIO.PWM(IN1,PWM1)
PWMAR = GPIO.PWM(IN2,PWM1)
PWMA.start(0)
PWMAR.start(0)

#Right Motor Setup
PWM2,IN3,IN4,IN5 = 100,17,18,23
GPIO.setup(IN4,GPIO.OUT) #Forwards
GPIO.setup(IN3,GPIO.OUT) #Reverse
PWMB = GPIO.PWM(IN4,PWM2)
PWMBR = GPIO.PWM(IN3,PWM2)
PWMB.start(0)
PWMBR.start(0)

#servo setup
GPIO.setup(IN5, GPIO.OUT)
servo = GPIO.PWM(23,50)
servo.start(0)
setspeed = int(input("SetSpeed: "))

# setup controls
def dpad(pos):
    if pos.top:
        PWMA.ChangeDutyCycle(setspeed)
        PWMB.ChangeDutyCycle(setspeed)
    elif pos.bottom:
        PWMAR.ChangeDutyCycle(setspeed)
        PWMBR.ChangeDutyCycle(setspeed)
    elif pos.left:
        PWMA.ChangeDutyCycle(setspeed)
        PWMBR.ChangeDutyCycle(setspeed)
    elif pos.right:
        PWMAR.ChangeDutyCycle(setspeed)
        PWMB.ChangeDutyCycle(setspeed)
    elif pos.middle:
        servo.ChangeDutyCycle(9)

def stop():
    PWMAR.ChangeDutyCycle(0)
    PWMBR.ChangeDutyCycle(0)
    PWMA.ChangeDutyCycle(0)
    PWMB.ChangeDutyCycle(0)
    servo.ChangeDutyCycle(6.75)

def swiped():
    print("exited cleanly")
    PWMA.stop()
    PWMB.stop()
    PWMBR.stop()
    PWMAR.stop()
    servo.stop()
    GPIO.cleanup()
       
bd.when_pressed = dpad
bd.when_moved = dpad
bd.when_swiped = swiped
bd.when_released = stop
pause()

