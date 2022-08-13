import numpy as np
# from scipy.spatial import distance
from sklearn.metrics import jaccard_score
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

def gen_discriptor_img(superpx_img, img, descr_func, descr_func_args=[None], descr_dims=3):
    descriptors = np.zeros((superpx_img.max()+1, descr_dims))
    im_descriptors = np.zeros((img.shape[0], img.shape[1], descr_dims), dtype=img.dtype)

    for i in range(superpx_img.min(), superpx_img.max()+1):
        args = [img, superpx_img==i] + descr_func_args
        descriptors[i] = descr_func(args)
        im_descriptors[superpx_img==i] = descriptors[i]

    return im_descriptors

####
#### Avg. RGB descriptor
def avg_rgb_descriptor(args):
    img = args[0]
    superpx_img_indicies = args[1]

    # Region as a column of rgb pairs:
    region = get_region1d(img, superpx_img_indicies)
    return [region[:,0].mean(), region[:,1].mean(), region[:,2].mean()]

####
#### Jaccard Similarity descriptor
def jaccard_descriptor(args):
    img = args[0]
    superpx_img_indicies = args[1]
    comp_img = args[2]

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

    region = np.float32(get_region1d(img, superpx_img_indicies))
    clusters_num = 3
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

    # Return the most dominant colour:
    return colours[0]

####
#### Mean and std. deviation of Hue and Saturation descriptor
def MAD(data):
    """
    Mean Absolute Deviation.
    """

    return np.mean(np.absolute(data - np.mean(data)))

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

####