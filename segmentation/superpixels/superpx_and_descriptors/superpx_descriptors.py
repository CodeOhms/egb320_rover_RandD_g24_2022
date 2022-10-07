import numpy as np
# from scipy.spatial import distance
# from sklearn.metrics import jaccard_score
from skimage.transform import resize
from skimage.util import img_as_ubyte
import cv2 as cv
import matplotlib.pyplot as plt

####
#### Generate descriptor images and regions
def get_region1d(img, superpx_img_indicies):
    return img[superpx_img_indicies]

def get_region2d(img, superpx_img_indicies):
    mask = np.zeros(img.shape[:2], dtype = "uint8")
    mask[superpx_img_indicies] = 255
    return cv.bitwise_and(img, img, mask=mask)

def gen_discriptor_img(superpx_img, img, descr_func, img_dtype=None, descr_func_args=[None], descr_dims=3):
    if img_dtype == None:
        img_dtype = img.dtype
    
    descriptors = [ ]
    im_descriptors = np.zeros((img.shape[0], img.shape[1], descr_dims), dtype=img_dtype)

    for i in range(superpx_img.min(), superpx_img.max()+1):
        args = [img, superpx_img==i, descr_dims] + descr_func_args
        descriptors.append(descr_func(args))
        im_descriptors[superpx_img==i] = descriptors[i]

    return im_descriptors

####
#### Avg. RGB descriptor
def avg_rgb_descriptor(args):
    img = args[0]
    superpx_img_indicies = args[1]

    # Region as a column of rgb pairs:
    region = get_region1d(img, superpx_img_indicies)
    return (region[:,0].mean(), region[:,1].mean(), region[:,2].mean())

####
#### Jaccard Similarity descriptor
def jaccard_descriptor(args):
    img = args[0]
    superpx_img_indicies = args[1]
    comp_img = args[3]

    region = get_region(img, superpx_img_indicies)
    # mask = np.zeros(img.shape[:2], dtype = "uint8")
    # mask[superpx_img_indicies] = 255
    # region = cv.bitwise_and(img, img, mask=mask)
    # Resize comp_img to region size if needed:
    if comp_img.shape != region.shape:
        comp_img = resize(comp_img.ravel(), region.shape)
        comp_img = img_as_ubyte(comp_img)
    return jaccard_score(comp_img.ravel(), region.ravel(), average="micro")

####
#### Dominant Colour descriptor
def dominant_colour_descriptor(args):
    img = args[0]
    superpx_img_indicies = args[1]
    clusters_num = args[3]

    region = np.float32(get_region1d(img, superpx_img_indicies))
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    flags = cv.KMEANS_RANDOM_CENTERS
    compactness, labels, centres = cv.kmeans(region, clusters_num, None, criteria, 10, flags)
    
    # Dominant colour counting method from here: https://medium.com/buzzrobot/dominant-colors-in-an-image-using-k-means-clustering-3c7af4622036
    #labels form 0 to no. of clusters
    numLabels = np.arange(0, clusters_num+1)
    
    #create frequency count tables    
    (hist, _) = np.histogram(labels, bins=numLabels)
    hist = hist.astype("float")
    hist /= hist.sum()
    
    #appending frequencies to cluster centers
    colours = centres
    
    #descending order sorting as per frequency count
    colours = colours[(-hist).argsort()]
    hist = hist[(-hist).argsort()]

    return colours.ravel()

####
#### Mean and mean absolute deviation of Hue and Saturation descriptor
def MAD(data):
    """
    Mean Absolute Deviation.
    """

    return np.mean(np.absolute(data - np.mean(data)))

def hs_stats_descriptor(args):
    img = args[0]
    superpx_img_indicies = args[1]

    img_f32 = np.float32(img)
    img_hsv = cv.cvtColor(img_f32, cv.COLOR_RGB2HSV_FULL)

    # Region as a column of HSV pairs:
    region = get_region1d(img_hsv, superpx_img_indicies)
    hue_avg = region[:,0].mean()
    sat_avg = region[:,1].mean()
    hue_mad = MAD(region[:,0])
    sat_mad = MAD(region[:,1])
    
    return (hue_avg, sat_avg, hue_mad, sat_mad)

def gen_hue_vectors(angles):
    c_hue_vec = [np.exp(complex(0,a)) for a in angles]
    return np.array( [[np.real(c_num),np.imag(c_num)] for c_num in c_hue_vec] )
def smallest_angle_between(angles0, angles1):
# https://stackoverflow.com/questions/7570808/how-do-i-calculate-the-difference-of-two-angle-measures
    phi = np.abs(angles1 - angles0) % 360
    distance = phi
    if phi > 180:
        distance = 360 - phi
    return distance
def MAD_from_hue(args):
    img = args[0]
    superpx_img_indicies = args[1]
    hue = args[3]

    img_f32 = np.float32(img)
    img_hsv = cv.cvtColor(img_f32, cv.COLOR_RGB2HSV_FULL)

    # Region as a column of HSV pairs:
    region = get_region1d(img_hsv, superpx_img_indicies)
    angles_between = np.array([smallest_angle_between(imgh,hue) for imgh in region[:,0]])
    hue_mad = np.mean(angles_between)

    return hue_mad
####