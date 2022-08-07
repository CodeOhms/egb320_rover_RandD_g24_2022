import numpy as np

####
#### Generate descriptor image
def gen_discriptor_img(superpx_img, img, descr_func):
    descriptors = np.zeros((superpx_img.max()+1,3))
    im_descriptors = np.zeros_like(img)

    for i in range(superpx_img.min(), superpx_img.max()+1):
        descriptors[i] = descr_func(img[superpx_img==i])
        im_descriptors[superpx_img==i] = descriptors[i]

    return im_descriptors

####
#### Avg. RGB descriptor
def avg_rgb_descriptor(region):
    return [region[:,0].mean(), region[:,1].mean(), region[:,2].mean()]

####