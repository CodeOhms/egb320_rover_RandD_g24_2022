import time
import numpy as np
import numexpr as ne
from imutils.video import VideoStream
from skimage import segmentation
#from fast_slic import Slic
from fast_slic.neon import SlicNeon as Slic
from skimage.segmentation import mark_boundaries
import cv2 as cv
# from scipy.spatial import distance
from scipy.spatial.distance import cdist
# import matplotlib.pyplot as plt

def superpx_slic_trans(img, num_regions):
    # slic = Slic(num_components=40, compactness=1, min_size_factor=0) # Supposedly gets FPS increase, but I don't see any...
    slic = Slic(num_components=num_regions, compactness=10)
    segments = slic.iterate(img) # Cluster Map
    return segments

def grid_superpx_trans(img, num_regions):
    n_cells_x = 15
    n_cells_y = 10
    size_x = img.shape[1]//n_cells_x
    size_y = img.shape[0]//n_cells_y
    print(size_x,size_y)

    segments = np.zeros(img.shape[:2], dtype=int)
    for y in range(n_cells_y):
        for x in range(n_cells_x):
            segments[y*size_y:(y+1)*size_y,x*size_x:(x+1)*size_x] = x + y*n_cells_x
    return segments

def gen_superpx_img(img, num_regions=40):
    return grid_superpx_trans(img, num_regions)

def get_region1d(img, superpx_img_indicies):
    return img[superpx_img_indicies]

def hue_sat_manhattan(args):
    img_comp = args[0]
    superpx_img_indicies = args[1]
    hue_sat_vec = args[3]

    # Region as a column of HSV pairs:
        # print('hue_mad_imgs shape')
        # print(hue_mad_imgs.shape)
    region = get_region1d(img_comp, superpx_img_indicies)
    mhdist_between = cdist(hue_sat_vec, region, metric='cityblock')
    hue_sat_q = np.quantile(mhdist_between, 0.25)

    return hue_sat_q

def gen_discriptor_img(superpx_img, img, descr_func, img_dtype=None, descr_func_args=[None], descr_dims=3):
    if img_dtype == None:
        img_dtype = img.dtype
    
    descriptors = [ ]
    im_descriptors = np.zeros((img.shape[0], img.shape[1], descr_dims), dtype=img_dtype)

    for i in range(superpx_img.min(), superpx_img.max()+1):
        args = [img, superpx_img==i, descr_dims] + descr_func_args
        descriptors.append(descr_func(args))
        im_descriptors[superpx_img==i] = descriptors[i]

    return (im_descriptors, descriptors)

def grab_frame(capture, res):
    frame = capture.read()
    return frame

loop = True
def on_close():
    global loop
    loop = False

def PoC(capture, cam_res):
    global loop

    num_classes = 3 # Sample, obstacle, rock

    sat_mids = [0.02, 0.73, 1.0]
    # Hues (in degrees): 2, 110, 207
    hue_mids = [0.03490658503988659154, 1.65806278939461309808, 3.56047167406843233692] # Sample, obstacle, rock
    sat_hue_cnums = ne.evaluate('sat_mids*exp(complex(0,hue_mids))')
    sat_hue_vecs = np.array([[hsm_comp.real, hsm_comp.imag] for hsm_comp in sat_hue_cnums])

    frame = grab_frame(capture, cam_res)

    prev_frame_time = 0
    new_frame_time = 0
    
    dbug_img_canvas = np.zeros((100,512,3),np.uint8)
    dbug_img = np.zeros((100,512,3),np.uint8)
    font = cv.FONT_HERSHEY_SIMPLEX

    empty_mask = np.zeros((frame.shape[:2]), dtype=np.uint8)
    masks_shape = np.concatenate((frame.shape[:2], np.array([num_classes])))
    masks_hue = np.zeros(masks_shape, dtype=np.uint8)
    masks_sat = np.zeros(masks_shape, dtype=np.uint8)
    masks_objs = np.zeros(masks_shape, dtype=np.uint8)
    frame_masked_objs = np.zeros(masks_shape)

    num_regions = 25
    while(loop):
        new_frame_time = time.time()
        
    # Capture frame from camera:
        frame = grab_frame(capture, cam_res)

    # Create superpixel image:
        superpx_img = gen_superpx_img(frame, num_regions)

    # Apply descriptors:
        frame_f32 = np.float32(frame)
        frame_hsv = cv.cvtColor(frame_f32, cv.COLOR_BGR2HSV_FULL)
        frame_hue = frame_hsv[:,:,0]
        frame_sat = frame_hsv[:,:,1]
        pi = np.pi
        frame_comp = ne.evaluate('frame_sat*exp(complex(0,frame_hue*pi/180))')
        frame_vecs = np.array([[[comp.real,comp.imag] for comp in row] for row in frame_comp[:]])

        # Hue MAD from given hue:
        hue_mads_imgs_and_decrs = np.array([gen_discriptor_img(superpx_img, frame_vecs, hue_sat_manhattan, descr_func_args=[np.array([ sat_hue_vecs[i] ])], descr_dims=1, img_dtype=np.float32)
            for i in range(num_classes)
        ])
        hue_mad_imgs = hue_mads_imgs_and_decrs[:,0]
        hue_mad_descrs = hue_mads_imgs_and_decrs[:,1]

    # Select descriptors to mask objects:
        mad_threshold = 0.15
        hue_mad_regions = np.zeros((1,3))
        hue_mad_labels = hue_mad_descrs
        for i_reg, reg_label in enumerate(np.unique(superpx_img)):
            hue_mad_regions = np.array([hmad_lab[i_reg] for hmad_lab in hue_mad_labels])
            reg_idxs = superpx_img == reg_label

            hue_img_idx = np.argmin(hue_mad_regions)
            print('region index', str(i_reg))
            print()
            if hue_mad_regions[hue_img_idx] < mad_threshold:
                print(hue_mad_regions[hue_img_idx])
                print()
                masks_hue[:,:,hue_img_idx][reg_idxs] = 255
            else:
                print('Skipped ', str(hue_mad_regions[hue_img_idx]))
                print()
    
    # # Find contours:
    #     contours, hierarchy = cv.findContours(mask_gb, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    # # Find the convex hull object for each contour:
    #     hull_list = []
    #     for i in range(len(contours)):
    #        hull = cv.convexHull(contours[i])
    #        hull_list.append(hull)

    # # Draw contours + hull results
    #     contours_img = np.copy(frame)
    #     cv.drawContours(contours_img, contours, -1, (5,255,0), 2) # Bright green
    #     cv.drawContours(contours_img, hull_list, -1, (250,0,255), 2) # Pink/purple

    # Display results:
        cv.imshow("Frame", frame)
        cv.imshow("Sample hue mask", masks_hue[:,:,0])
        cv.imshow("Obstacle hue mask", masks_hue[:,:,1])
        cv.imshow("Rock hue mask", masks_hue[:,:,2])
        # cv.imshow("Sample hue mask", masks_objs[:,:,0])
        # cv.imshow("Obstacle hue mask", masks_objs[:,:,1])
        # cv.imshow("Rock hue mask", masks_objs[:,:,2])
        # cv.imshow("Saturation mask", mask_sat_gb)
        # cv.imshow("Superpixels", mark_boundaries(frame, superpx_img))
        # cv.imshow("Masked", masked_gb)
        # cv.imshow("Contours", contours_img)
        key = cv.waitKey(1) & 0xFF
        #if the `q` key was pressed, break from the loop
        if key == ord("q"):
            on_close()
            
        fps = 1/(new_frame_time - prev_frame_time)
        cv.putText(dbug_img, "FPS "+str(fps), (0,25), font, 1, (255,255,255), 2, cv.LINE_AA)
        cv.imshow("Debug message", dbug_img)
        prev_frame_time = new_frame_time
        dbug_img = np.copy(dbug_img_canvas)
        masks_hue[:,:,0] = np.copy(empty_mask)
        masks_hue[:,:,1] = np.copy(empty_mask)
        masks_hue[:,:,2] = np.copy(empty_mask)

def init_camera(camera):
# https://picamera.readthedocs.io/en/release-1.13/recipes1.html#capturing-consistent-images
    # Set ISO to the desired value
    camera.iso = 800
    # Wait for the automatic gain control to settle
    time.sleep(2)
    # Now fix the values
    camera.shutter_speed = camera.exposure_speed
    camera.exposure_mode = 'off'
    g = camera.awb_gains
    camera.awb_mode = 'off'
    camera.awb_gains = g

if __name__ == "__main__":
    # initialize the video stream and allow the cammera sensor to warmup
    # Vertical res must be multiple of 16, and horizontal a multiple of 32
    cam_res = (128, 64)
    video_stream = VideoStream(usePiCamera=True, resolution=cam_res, framerate=20, ).start()
    print("Camera warming up...")
    init_camera(video_stream.camera)
    
    PoC(video_stream, cam_res)
    
    cv.destroyAllWindows()
    video_stream.stop()
    
    print("Exiting.")
