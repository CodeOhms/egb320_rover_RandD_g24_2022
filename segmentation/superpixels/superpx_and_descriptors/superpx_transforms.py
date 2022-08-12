from skimage import filters
from skimage.morphology import disk
from skimage.feature import peak_local_max
from skimage import segmentation
from skimage import measure


####
#### Watershed
def watershed_get_markers(im, guass_sigma=3, sobel_struc_elem=disk(5), local_maxima_params=(0.2, 0, False)):
    min_dist = int(im.shape[1]/16)
    print('min_dist', int(min_dist))
    im_ = filters.gaussian(im, sigma=guass_sigma)
    gradg = filters.rank.gradient(im_[:,:,1], sobel_struc_elem).astype('int')
    gradb = filters.rank.gradient(im_[:,:,2], sobel_struc_elem).astype('int')
    gradr = filters.rank.gradient(im_[:,:,0], sobel_struc_elem).astype('int')
    grad = gradr+gradg+gradb
    
    return peak_local_max(grad.max()-grad,threshold_rel=local_maxima_params[0], min_distance=min_dist, indices=local_maxima_params[2]),grad

def superpx_watershed_trans(grad_img, markers):
    markers = measure.label(markers)
    superpx_im = segmentation.watershed(grad_img, markers)
    return superpx_im

####
#### SLIC
def superpx_slic_trans(img):
    superpx_im = segmentation.slic(img, slic_zero=True)
    return superpx_im
    
####