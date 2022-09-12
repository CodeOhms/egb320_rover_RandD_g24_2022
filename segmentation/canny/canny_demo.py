# https://riptutorial.com/opencv/example/26315/canny-edge-video-from-webcam-capture---python

import time
from imutils.video import VideoStream
import cv2

def canny_webcam():
    "Live capture frames from webcam and show the canny edge image of the captured frames."

    # cap = cv2.VideoCapture(0)
    cam_res = (128, 64)
    cap = VideoStream(usePiCamera=True, resolution=cam_res, framerate=20).start()
    print("Camera warming up...")
    time.sleep(2.0)

    while True:
        # ret, frame = cap.read()  # ret gets a boolean value. True if reading is successful (I think). frame is an
        frame = cap.read()
        # uint8 numpy.ndarray

        frame = cv2.GaussianBlur(frame, (7, 7), 2)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        edge = cv2.Canny(frame, 30, 170)

        cv2.imshow('Canny Edge', edge)

        if cv2.waitKey(20) == ord('q'):  # Introduce 20 milisecond delay. press q to exit.
            break
    
    cap.stop()

canny_webcam()