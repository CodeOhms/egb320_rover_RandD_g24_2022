import numpy as np
from skimage import segmentation
from skimage.segmentation import mark_boundaries
import cv2 as cv
import matplotlib.pyplot as plt

def superpx_slic_trans(img):
    superpx_im = segmentation.slic(img, slic_zero=True)
    return superpx_im

def gen_superpx_img(img):
    return superpx_slic_trans(img)

def MAD(data):
    """
    Mean Absolute Deviation.
    """

    return np.mean(np.absolute(data - np.mean(data)))

def get_region1d(img, superpx_img_indicies):
    return img[superpx_img_indicies]

def hs_stats_descriptor(args):
    img = args[0]
    superpx_img_indicies = args[1]

    img_hsv = cv.cvtColor(img, cv.COLOR_RGB2HSV)

    # Region as a column of HSV pairs:
    region = get_region1d(img_hsv, superpx_img_indicies)
    hue_avg = region[:,0].mean()
    sat_avg = region[:,1].mean()
    hue_mad = MAD(region[:,0])
    sat_mad = MAD(region[:,1])

    return (hue_avg, sat_avg, hue_mad, sat_mad)

def gen_discriptor_img(superpx_img, img, descr_func, descr_func_args=[None], descr_dims=3):
    descriptors = np.zeros((superpx_img.max()+1, descr_dims))
    im_descriptors = np.zeros((img.shape[0], img.shape[1], descr_dims), dtype=img.dtype)

    for i in range(superpx_img.min(), superpx_img.max()+1):
        args = [img, superpx_img==i] + descr_func_args
        descriptors[i] = descr_func(args)
        im_descriptors[superpx_img==i] = descriptors[i]

    return im_descriptors

def grab_frame(capture):
    ret, frame = capture.read()
    return cv.cvtColor(frame, cv.COLOR_BGR2RGB)

loop = True
def on_close(event):
    global loop
    loop = False

if __name__ == "__main__":
    capture = cv.VideoCapture(-1)

    hue_range_gb = np.array([[0, 38], [155, 180]], dtype=np.uint8)
    sat_deviation = 0.15
    # sat_range_gb = np.array([round(255*(0.81-sat_deviation)), round(255*(0.81+sat_deviation))], dtype=np.uint8)
    sat_range_gb = np.array([128, 255], dtype=np.uint8)

    frame = grab_frame(capture)

    frame_fig = plt.figure()
    frame_imgobj = plt.imshow(frame)
    gb_masked_fig = plt.figure()
    gb_masked_fig.canvas.mpl_connect('close_event', on_close)
    gb_masked_imgobj = plt.imshow(frame)
    plt.ion()

    while(loop):
    # Capture frame from camera:
        frame = grab_frame(capture)

    # Create superpixel image:
        superpx_img = gen_superpx_img(frame)

    # Apply descriptors:
        descr_img = gen_discriptor_img(superpx_img, frame, hs_stats_descriptor, descr_dims=4)

    # Select descriptors to mask objects:
        mask_hue_gb = cv.bitwise_or(cv.inRange(descr_img[:,:,0], int(hue_range_gb[0,0]), int(hue_range_gb[0,1])),
            cv.inRange(descr_img[:,:,0], int(hue_range_gb[1,0]), int(hue_range_gb[1,1])))
        mask_gb = cv.bitwise_and(mask_hue_gb, cv.inRange(descr_img[:,:,1], int(sat_range_gb[0]), int(sat_range_gb[1])))
        masked_gb = cv.bitwise_and(frame, frame, mask=mask_gb)

    # Display results:
        frame_imgobj.set_data(mark_boundaries(frame, superpx_img))
        frame_fig.canvas.draw()
        frame_fig.canvas.flush_events()
        gb_masked_imgobj.set_data(masked_gb)
        gb_masked_fig.canvas.draw()
        gb_masked_fig.canvas.flush_events()
        plt.pause(0.005)
    
    plt.show()
    plt.ioff()
