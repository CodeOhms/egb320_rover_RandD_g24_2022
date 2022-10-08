import os
from inspect import getsourcefile
import time
import copy
import numpy as np
import numexpr as ne
from imutils.video import VideoStream
from fast_slic import Slic
# from fast_slic.neon import SlicNeon as Slic
import cv2 as cv
from scipy.spatial.distance import cdist

def superpx_slic_trans(img, num_regions=40):
    slic = Slic(num_components=num_regions, compactness=10, min_size_factor=0) # Supposedly gets FPS increase, but I don't see any...
    # slic = Slic(num_components=num_regions, compactness=10)
    segments = slic.iterate(img) # Cluster Map
    return segments

def grid_superpx_trans(img, regions_props):
    n_cells_x = regions_props[0]
    n_cells_y = regions_props[1]
    size_x = regions_props[2]
    size_y = regions_props[3]
    
    segments = np.zeros(img.shape[:2], dtype=int)
    for y in range(n_cells_y):
        for x in range(n_cells_x):
            segments[y*size_y:(y+1)*size_y,x*size_x:(x+1)*size_x] = x + y*n_cells_x
    return segments

def gen_superpx_img(img, regions_props):
    return superpx_slic_trans(img, regions_props)
    # return grid_superpx_trans(img, regions_props) # Slower without multithreading...

def get_region1d(img, superpx_img_indicies):
    return img[superpx_img_indicies]

def hue_sat_manhattan(args):
    img_comp = args[0]
    superpx_img_indicies = args[1]
    hue_sat_vec = args[3]

    # Region as a column of HSV pairs:
    region = get_region1d(img_comp, superpx_img_indicies)
    mhdist_between = cdist(hue_sat_vec, region, metric='cityblock')
    # hue_sat_q = np.quantile(mhdist_between, 0.25)
    hue_sat_q = np.mean(mhdist_between)

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

def PoC(capture, cam_res, imgs_dir):
    global loop

    num_classes = 6 # Wall and floor, sample, obstacle, rock, lander

    sat_mids = [0, 0, 0.65, 0.85, 0.95, 1.0]
    # Hues (in degrees): 180, 220, 2, 110, 207
    hue_mids = [
        3.14159265358979323846, 3.83972435438752506923, 0.03490658503988659154,
        1.65806278939461309808, 3.56047167406843233692, 1.04719755119659774615
    ] # Wall and floor, sample, obstacle, rock, lander
    sat_hue_cnums = ne.evaluate('sat_mids*exp(complex(0,hue_mids))')
    sat_hue_vecs = np.array([[hsm_comp.real, hsm_comp.imag] for hsm_comp in sat_hue_cnums])

    frame = grab_frame(capture, cam_res)
    f_scale = 4

    f_height = frame.shape[0]
    f_width = frame.shape[1]

    prev_frame_time = 0
    new_frame_time = 0
    
    font = cv.FONT_HERSHEY_SIMPLEX


    num_regions = 25
    regions_properties = [16, 8, 0, 0] # n_cells_x, n_cells_y, size_x, size_y
    regions_shape = [f_width//regions_properties[0], f_height//regions_properties[1]]
    regions_properties[2:] = regions_shape

    while(loop):
        new_frame_time = time.time()

    # Setup display images:
        masks_shape = np.concatenate((frame.shape[:2], np.array([num_classes])))
        masks_hue = np.zeros(masks_shape, dtype=np.uint8)
        masks_sat = np.zeros(masks_shape, dtype=np.uint8)
        masks_objs = np.zeros(masks_shape, dtype=np.uint8)
        frame_masked_objs = np.zeros(masks_shape)
        dbug_img = np.zeros((100,512,3),np.uint8)
        f_upscale = cv.resize(copy.deepcopy(frame), dsize=None, fx=f_scale, fy=f_scale, interpolation= cv.INTER_LINEAR) # Upscale for display!
        contours_imgs = [copy.deepcopy(f_upscale) for x in range(num_classes)]
        
    # Capture frame from camera:
        frame = grab_frame(capture, cam_res)

    # Create superpixel image:
        superpx_img = gen_superpx_img(frame, num_regions)
        # superpx_img = gen_superpx_img(frame, regions_properties)

    # Apply descriptors:
        frame_f32 = np.float32(frame)
        frame_hsv = cv.cvtColor(frame_f32, cv.COLOR_BGR2HSV_FULL)
        frame_hue = frame_hsv[:,:,0]
        frame_sat = frame_hsv[:,:,1]
        pi = np.pi
        frame_comp = ne.evaluate('frame_sat*exp(complex(0,frame_hue*pi/180))')
        frame_vecs = np.array([[[comp.real,comp.imag] for comp in row] for row in frame_comp[:]])

        # Hue MAD from given hue:
        hue_mads_imgs_and_decrs = np.array(
            [gen_discriptor_img(superpx_img, frame_vecs, hue_sat_manhattan,
                                descr_func_args=[np.array([ sat_hue_vecs[i] ])], descr_dims=1, img_dtype=np.float32)
                                for i in range(num_classes)
            ]
        )
        hue_mad_imgs = hue_mads_imgs_and_decrs[:,0]
        hue_mad_descrs = hue_mads_imgs_and_decrs[:,1]

    # Select descriptors to mask objects:
        # mad_threshold = 0.15
        mad_threshold = 1.0
        hue_mad_regions = np.zeros((1,3))
        hue_mad_labels = hue_mad_descrs
        for i_reg, reg_label in enumerate(np.unique(superpx_img)):
            hue_mad_regions = np.array([hmad_lab[i_reg] for hmad_lab in hue_mad_labels])
            reg_idxs = superpx_img == reg_label

            hue_img_idx = np.argmin(hue_mad_regions)
            # print('region index', str(i_reg))
            # print()
            if hue_mad_regions[hue_img_idx] < mad_threshold:
                # print(hue_mad_regions[hue_img_idx])
                # print()
                masks_hue[:,:,hue_img_idx][reg_idxs] = 255
            # else:
            #     print('Skipped ', str(hue_mad_regions[hue_img_idx]))
            #     print()
    
    # Find contours:
        hulls_lists = [[ ] for x in range(num_classes)]
        contours = [ ]
        for m_i in range(num_classes):
            conts, _ = cv.findContours(masks_hue[:,:,m_i], cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
            contours.append(conts)
        # Find the convex hull object for each contour, and draw hulls:
            for cnt in conts:
                hulls = cv.convexHull(cnt)
                hulls_lists[m_i].append(hulls)
            cv.drawContours(contours_imgs[m_i], np.multiply(f_scale, hulls_lists[m_i]), -1, (250,0,255), 2) # magenta
    
    # Correct for distortions:

    # Estimate bearing:
        fov = [62.2, 48.8]
        moments = [ ]
        cen_x = [[ ] for x in range(num_classes)]
        bearings = [[ ] for x in range(num_classes)]
        for ih_i, img_hulls in enumerate(hulls_lists):
            for hull in img_hulls:
                # try: # may not be any detected contours!
                moments.append(cv.moments(hull))
                try: # In case of division by zero:
                    cx = int(moments[-1]['m10']/moments[-1]['m00'])
                    cy = int(moments[-1]['m01']/moments[-1]['m00'])
                except:
                    pass
                scaled_cen = (cx*f_scale,cy*f_scale)
                contours_imgs[ih_i] = cv.circle(contours_imgs[ih_i], scaled_cen, radius=1, color=(0, 0, 255), thickness=-1)
                cen_x[ih_i].append(cx)
                bearing = (cx-f_width/2)/f_width*fov[0]
                bearings[ih_i].append(bearing)
                cv.putText(contours_imgs[ih_i], "B: "+str(bearing), scaled_cen, font, 0.3, (255,255,255), 1, cv.LINE_AA)

    # Estimate distance:
        

    # Display results:
        cv.imshow("Frame", frame)
        cv.imshow("Sample mask", masks_hue[:,:,2])
        cv.imshow("Obstacle mask", masks_hue[:,:,3])
        cv.imshow("Rock mask", masks_hue[:,:,4])
        cv.imshow("Lander mask", masks_hue[:,:,5])
        cv.imshow("Sample conv. hulls", contours_imgs[2])
        cv.imshow("Obstacle conv. hulls", contours_imgs[3])
        cv.imshow("Rock conv. hulls", contours_imgs[4])
        cv.imshow("Lander conv. hulls", contours_imgs[5])
        key = cv.waitKey(1) & 0xFF
        #if the `q` key was pressed, break from the loop
        if key == ord("q"):
            on_close()
            
        fps = 1/(new_frame_time - prev_frame_time)
        cv.putText(dbug_img, "FPS "+str(fps), (0,25), font, 1, (255,255,255), 2, cv.LINE_AA)
        cv.imshow("Debug message", dbug_img)
        prev_frame_time = new_frame_time

def init_camera(camera):
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

if __name__ == "__main__":
    # Set the working directory to the root folder of the R&D git repo:
    script_dir = os.path.dirname(getsourcefile(lambda:0))
    os.chdir(script_dir)
    os.chdir('../../../')
    imgs_dir = 'segmentation/superpixels/superpx_PoC/'

    # initialize the video stream and allow the cammera sensor to warmup
    # Vertical res must be multiple of 16, and horizontal a multiple of 32
    cam_res = (64, 32)
    video_stream = VideoStream(usePiCamera=True, resolution=cam_res, rotation=180).start()
    print("Camera warming up...")
    init_camera(video_stream.camera)
    
    PoC(video_stream, cam_res, imgs_dir)
    
    cv.destroyAllWindows()
    video_stream.stop()
    
    print("Exiting.")
