# Inspired from https://github.com/adfoucart/image-processing-notebooks/blob/main/V31%20-%20Region%20growing%20with%20the%20watershed%20transform.ipynb

from matplotlib import pyplot as plt
import numpy as np
from skimage.io import imread,imsave

from inspect import getsourcefile
import os
script_dir = os.path.dirname(getsourcefile(lambda:0))
os.chdir(script_dir)
os.chdir('../../../')
print(os.getcwd())

im = imread('test_media/imgs/test_img_0.jpg')
print(im.shape)

plt.figure()
plt.imshow(im)
plt.show()

from skimage.filters import rank,gaussian
from skimage.morphology import disk
from skimage.feature import peak_local_max

def get_markers(im, indices=False):
    im_ = gaussian(im, sigma=3)
    gradr = rank.gradient(im_[:,:,0],disk(5)).astype('int')
    gradg = rank.gradient(im_[:,:,1],disk(5)).astype('int')
    gradb = rank.gradient(im_[:,:,2],disk(5)).astype('int')
    grad = gradr+gradg+gradb
    
    # from skimage.color import rgb2hsv
    # im_hsv = rgb2hsv(im)
    # im_ = gaussian(im_hsv, sigma=3)
    # grad_h = rank.gradient(im_[:,:,0],disk(5)).astype('int')
    # grad_s = rank.gradient(im_[:,:,1],disk(5)).astype('int')
    # grad = grad_h + grad_s
    
    return peak_local_max(grad.max()-grad,threshold_rel=0.2, min_distance=250,indices=indices),grad

markers,grad = get_markers(im, True)
plt.figure()
plt.imshow(grad, cmap=plt.cm.gray)
plt.plot(markers[:,1],markers[:,0],'b+')
plt.show()

# from skimage.morphology import watershed
from skimage.segmentation import watershed
from skimage.segmentation import mark_boundaries
from skimage.measure import label

markers, grad = get_markers(im, False)
markers = label(markers)
ws = watershed(grad, markers)

plt.figure()
plt.imshow(mark_boundaries(im,ws))
plt.figure()
plt.imshow(ws)
plt.show()
