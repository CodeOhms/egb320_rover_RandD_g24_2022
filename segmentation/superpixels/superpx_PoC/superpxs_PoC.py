import time
import numpy as np
from imutils.video import VideoStream
from skimage import segmentation
#from fast_slic import Slic
from fast_slic.neon import SlicNeon as Slic
from skimage.segmentation import mark_boundaries
import cv2 as cv
import matplotlib.pyplot as plt

def superpx_slic_trans(img):
    # slic = Slic(num_components=40, compactness=1, min_size_factor=0) # Supposedly gets FPS increase, but I don't see any...
    slic = Slic(num_components=40, compactness=10)
    assignment = slic.iterate(img) # Cluster Map
    return assignment

def gen_superpx_img(img):
    return superpx_slic_trans(img)

def get_region1d(img, superpx_img_indicies):
    return img[superpx_img_indicies]

def hs_stats_descriptor(args):
    img = args[0]
    superpx_img_indicies = args[1]

    img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    # Region as a column of HSV pairs:
    region = get_region1d(img_hsv, superpx_img_indicies)
    hue_avg = region[:,0].mean()
    sat_avg = region[:,1].mean()

    return (hue_avg, sat_avg)

def gen_discriptor_img(superpx_img, img, descr_func, descr_func_args=[None], descr_dims=3):
    descriptors = np.zeros((superpx_img.max()+1, descr_dims))
    im_descriptors = np.zeros((img.shape[0], img.shape[1], descr_dims), dtype=img.dtype)

    for i in range(superpx_img.min(), superpx_img.max()+1):
        args = [img, superpx_img==i] + descr_func_args
        descriptors[i] = descr_func(args)
        im_descriptors[superpx_img==i] = descriptors[i]

    return im_descriptors

def grab_frame(capture, res):
    frame = capture.read()
    return frame

loop = True
def on_close():
    global loop
    loop = False

def PoC(capture, cam_res):
    global loop
    
    hue_range_gb = np.array([[0, 38], [150, 180]])
    sat_deviation = 0.15
    sat_mid_gb = 0.8125
    sat_range_gb = np.array([round(255*(sat_mid_gb-sat_deviation)), round(255*(sat_mid_gb+sat_deviation))], dtype=np.uint8)
    # sat_range_gb = np.array([128, 255], dtype=np.uint8)

    frame = grab_frame(capture, cam_res)

    prev_frame_time = 0
    new_frame_time = 0
    
    dbug_img_canvas = np.zeros((100,512,3),np.uint8)
    dbug_img = np.zeros((100,512,3),np.uint8)
    font = cv.FONT_HERSHEY_SIMPLEX

    while(loop):
        new_frame_time = time.time()
        
    # Capture frame from camera:
        frame = grab_frame(capture, cam_res)

    # Create superpixel image:
        superpx_img = gen_superpx_img(frame)

    # Apply descriptors:
        descr_img = gen_discriptor_img(superpx_img, frame, hs_stats_descriptor, descr_dims=2)

    # Select descriptors to mask objects:
        mask_hue_gb = cv.bitwise_or(cv.inRange(descr_img[:,:,0], int(hue_range_gb[0,0]), int(hue_range_gb[0,1])),
            cv.inRange(descr_img[:,:,0], int(hue_range_gb[1,0]), int(hue_range_gb[1,1])))
        mask_sat_gb = cv.inRange(descr_img[:,:,1], int(sat_range_gb[0]), int(sat_range_gb[1]))
        mask_gb = cv.bitwise_and(mask_hue_gb, mask_sat_gb)
        masked_gb = cv.bitwise_and(frame, frame, mask=mask_gb)
    
    # Find contours:
        contours, hierarchy = cv.findContours(mask_gb, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    # Find the convex hull object for each contour:
        hull_list = []
        for i in range(len(contours)):
           hull = cv.convexHull(contours[i])
           hull_list.append(hull)

    # Draw contours + hull results
        contours_img = np.copy(frame)
        cv.drawContours(contours_img, contours, -1, (5,255,0), 2) # Bright green
        cv.drawContours(contours_img, hull_list, -1, (250,0,255), 2) # Pink/purple

    # Display results:
        cv.imshow("Frame", frame)
        cv.imshow("Hue mask", mask_hue_gb)
        cv.imshow("Saturation mask", mask_sat_gb)
        cv.imshow("Superpixels", mark_boundaries(frame, superpx_img))
        cv.imshow("Masked", masked_gb)
        cv.imshow("Contours", contours_img)
        key = cv.waitKey(1) & 0xFF
        #if the `q` key was pressed, break from the loop
        if key == ord("q"):
            on_close()
            
        fps = 1/(new_frame_time - prev_frame_time)
        cv.putText(dbug_img, "FPS "+str(fps), (0,25), font, 1, (255,255,255), 2, cv.LINE_AA)
        cv.imshow("Debug message", dbug_img)
        prev_frame_time = new_frame_time
        dbug_img = np.copy(dbug_img_canvas)
    
if __name__ == "__main__":
    # initialize the video stream and allow the cammera sensor to warmup
    # Vertical res must be multiple of 16, and horizontal a multiple of 32
    cam_res = (128, 64)
    video_stream = VideoStream(usePiCamera=True, resolution=cam_res, framerate=20).start()
    print("Camera warming up...")
    time.sleep(2.0)
    
    PoC(video_stream, cam_res)
    
    cv.destroyAllWindows()
    video_stream.stop()
    
    print("Exiting.")
