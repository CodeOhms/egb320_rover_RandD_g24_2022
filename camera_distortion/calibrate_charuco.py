# https://medium.com/vacatronics/3-ways-to-calibrate-your-camera-using-opencv-and-python-395528a51615
# Calibration board image from: https://docs.opencv.org/3.4/charucoboard.png
# import os
# from inspect import getsourcefile
from charuco import calibrate_charuco
from utils import load_coefficients, save_coefficients
import cv2

# # Get the working directory:
# script_dir = os.path.dirname(getsourcefile(lambda:0))

# Parameters
IMAGES_DIR = 'calib_imgs/'
# IMAGES_DIR = 'path_to_images'
IMAGES_FORMAT = 'png'
# Dimensions in cm
MARKER_LENGTH = 2.2
SQUARE_LENGTH = 3.6


# Calibrate 
ret, mtx, dist, rvecs, tvecs = calibrate_charuco(
    IMAGES_DIR, 
    IMAGES_FORMAT,
    MARKER_LENGTH,
    SQUARE_LENGTH
)
# Save coefficients into a file
save_coefficients(mtx, dist, "calibration_charuco.yml")

# Load coefficients
mtx, dist = load_coefficients('calibration_charuco.yml')
original = cv2.imread(IMAGES_DIR+'calib_img_2022-10-09_1003.png')
# original = cv2.imread('test_img_2.jpg')
h, w = original.shape[:2]
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 0, (w,h))
mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w,h), 5)
dst = cv2.remap(original, mapx, mapy, cv2.INTER_LINEAR)
# dst = cv2.undistort(original, mtx, dist, None, mtx)
cv2.imwrite('undist_charuco.jpg', dst)

