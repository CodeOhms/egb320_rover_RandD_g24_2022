import numpy as np
from imutils.video import VideoStream
import cv2 as cv

def init_picamera(camera):
# https://picamera.readthedocs.io/en/release-1.13/recipes1.html#capturing-consistent-images
    # Set ISO to the desired value
    camera.iso = 900
    # Wait for the automatic gain control to settle
    time.sleep(2)
    # Now fix the values
    camera.shutter_speed = camera.exposure_speed
    camera.exposure_mode = 'off'
    g = camera.awb_gains
    camera.awb_mode = 'off'
    camera.awb_gains = g

def init_usbcamera(ocv_vid, cam_res):
    backend = ocv_vid.get(cv.CAP_PROP_BACKEND)
    print('OpenCV camera backend', backend)
    print()
    
    f_width = ocv_vid.get(cv.CAP_PROP_FRAME_WIDTH)
    f_height = ocv_vid.get(cv.CAP_PROP_FRAME_HEIGHT)
    iso = ocv_vid.get(cv.CAP_PROP_ISO_SPEED)
    exp = ocv_vid.get(cv.CAP_PROP_EXPOSURE)
    aexp = ocv_vid.get(cv.CAP_PROP_AUTO_EXPOSURE)
    print('Props before:')
    print('Frame width', f_width)
    print('Frame height', f_height)
    print('ISO', iso)
    print('Exposure', exp)
    print('Auto exposure', aexp)
    print()
    
    ocv_vid.set(cv.CAP_PROP_FRAME_WIDTH, cam_res[0])
    ocv_vid.set(cv.CAP_PROP_FRAME_HEIGHT, cam_res[1])
    
    f_width = ocv_vid.get(cv.CAP_PROP_FRAME_WIDTH)
    f_height = ocv_vid.get(cv.CAP_PROP_FRAME_HEIGHT)
    
    print('Props after:')
    print('Frame width', f_width)
    print('Frame height', f_height)
    print()
    

def grab_frame(capture, res):
    ret, frame = capture.read()
    #frame = capture.read()
    return frame

def cam_test(cap, cam_res):
    loop = True
    while(loop):
        frame = grab_frame(cap, cam_res)
        asdf = np.zeros((cam_res[1], cam_res[0]))
        cv.imshow('Frame', frame)
        key = cv.waitKey(1) & 0xFF
        if key == ord("q"):
            loop = False

if __name__ == "__main__":
    # initialize the video stream and allow the cammera sensor to warmup
    # Vertical res must be multiple of 16, and horizontal a multiple of 32
    cam_res = (64, 32)
    #video_stream = VideoStream(usePiCamera=True, resolution=cam_res, rotation=180).start()
    video_stream = cv.VideoCapture(0)
    if not video_stream.isOpened():
        print('Unable to open the camera!')
    else:
        print("Camera warming up...")
        init_usbcamera(video_stream, cam_res)
        
        cam_test(video_stream, cam_res)
        
        cv.destroyAllWindows()
    video_stream.release()
    #video_stream.stop()
