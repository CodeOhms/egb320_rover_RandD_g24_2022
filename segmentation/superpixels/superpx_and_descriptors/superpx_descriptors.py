import numpy as np
# from scipy.spatial import distance
from sklearn.metrics import jaccard_score
from skimage.transform import resize
from skimage.util import img_as_ubyte
import cv2 as cv
import matplotlib.pyplot as plt

####
#### Generate descriptor image
def gen_discriptor_img(superpx_img, img, descr_func, descr_func_args=[None]):
    descriptors = np.zeros((superpx_img.max()+1,3))
    im_descriptors = np.zeros_like(img)

    # for (i, segVal) in enumerate(np.unique(superpx_img)):
    #     mask = np.zeros(img.shape[:2], dtype = "uint8")
    #     mask[superpx_img == segVal] = 255
    #     print(i, segVal)
    #     plt.figure()
    #     plt.imshow(mask)
    #     plt.figure()
    #     plt.imshow(img[superpx_img==i])
    #     plt.show()

    # for i in range(superpx_img.min(), superpx_img.max()+1):
    #     args = [img[superpx_img==i]] + descr_func_args
    #     plt.imshow(img[superpx_img==i])
    #     plt.show()
    #     descriptors[i] = descr_func(args)
    #     im_descriptors[superpx_img==i] = descriptors[i]

    # for i in range(superpx_img.min(), superpx_img.max()+1):
    #     mask = np.zeros(img.shape[:2], dtype = "uint8")
    #     mask[superpx_img==i] = 255
    #     args = [cv.bitwise_and(img, img, mask=mask)] + descr_func_args
    #     descriptors[i] = descr_func(args)
    #     im_descriptors[superpx_img==i] = descriptors[i]

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
    region = img[superpx_img_indicies]
    return [region[:,0].mean(), region[:,1].mean(), region[:,2].mean()]

####
#### Jaccard Similarity descriptor
def jaccard_descriptor(args):
    img = args[0]
    superpx_img_indicies = args[1]
    comp_img = args[2]

    mask = np.zeros(img.shape[:2], dtype = "uint8")
    mask[superpx_img_indicies] = 255
    region = cv.bitwise_and(img, img, mask=mask)
    # Resize comp_img to region size if needed:
    if comp_img.shape != region.shape:
        comp_img = resize(comp_img.ravel(), region.shape)
        comp_img = img_as_ubyte(comp_img)
    return jaccard_score(comp_img.ravel(), region.ravel(), average="micro")

####