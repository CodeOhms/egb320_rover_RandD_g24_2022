from skimage import filters
from skimage.morphology import disk
from skimage.feature import peak_local_max
from skimage import segmentation
from skimage.segmentation import mark_boundaries
from fast_slic import Slic
from skimage import measure

####
#### Helpers
def show_boundaries(image, suppx_img, output_scale=255, output_channels=slice(0, 1), output_dtype=None):
    marked_img = mark_boundaries(image, suppx_img, color=[0,0,0])
    idx = None
    if output_channels.stop - output_channels.start == 1:
        idx = output_channels.start
    else:
        idx = output_channels
    marked_img = marked_img[:, :, idx]
    marked_img *= output_scale
    if output_dtype is not None:
        marked_img = marked_img.astype(output_dtype)
    
    return marked_img


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
    # superpx_im = segmentation.slic(img, slic_zero=True)
    # return superpx_im

    # Try a potentially faster SLIC implementation:
    slic = Slic(num_components=50, compactness=10)
    assignment = slic.iterate(img) # Cluster Map
    return assignment
    
####