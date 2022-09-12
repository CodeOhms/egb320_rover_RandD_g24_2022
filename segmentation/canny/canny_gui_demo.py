# https://stackoverflow.com/questions/25125670/best-value-for-threshold-in-canny

import time
from imutils.video import VideoStream
import cv2
import numpy as np 
import matplotlib.pyplot as plt 

def callback(x):
    print(x)

# img = cv2.imread('your_image.png', 0) #read image as grayscale
cam_res = (128, 64)
cap = VideoStream(usePiCamera=True, resolution=cam_res, framerate=20).start()
print("Camera warming up...")
time.sleep(1)


canny = np.zeros((64, 128), dtype=np.uint8)

cv2.namedWindow('image') # make a window with name 'image'
cv2.createTrackbar('L', 'image', 0, 255, callback) #lower threshold trackbar for window 'image
cv2.createTrackbar('U', 'image', 0, 255, callback) #upper threshold trackbar for window 'image

while(1):
    img = cap.read()
    img = cv2.GaussianBlur(img, (7, 7), 2)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    numpy_horizontal_concat = np.concatenate((img, canny), axis=1) # to display image side by side
    cv2.imshow('image', numpy_horizontal_concat)
    k = cv2.waitKey(1) & 0xFF
    if k == 27: #escape key
        break
    l = cv2.getTrackbarPos('L', 'image')
    u = cv2.getTrackbarPos('U', 'image')

    canny = cv2.Canny(img, l, u)

cap.stop()
cv2.destroyAllWindows()