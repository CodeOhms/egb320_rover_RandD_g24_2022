from inspect import getsourcefile
import os
import math
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.segmentation import mark_boundaries

from superpx_transforms import *
from superpx_descriptors import *

if __name__ == '__main__':
    # Set the working directory to the root folder of the R&D git repo:
    script_dir = os.path.dirname(getsourcefile(lambda:0))
    os.chdir(script_dir)
    os.chdir('../../../')
    imgs_dir = 'test_media/imgs/'

    img = imread(imgs_dir+'test_img_0.jpg')

    # Apply superpixel algos:
    superpx_img = None

        # Watershed:
    ws_markers, rgb_grad_im = watershed_get_markers(img)
    superpx_img = superpx_watershed_trans(rgb_grad_im, ws_markers)

    descr_imgs = [ ]
    # Avg. RGB:
    descr_imgs.append(gen_discriptor_img(superpx_img, img, avg_rgb_descriptor))

    # Jaccard Similarity:
    # sample_comp_img = imread(imgs_dir+'sample_jacc_comp.png')
    # descr_imgs.append(gen_discriptor_img(superpx_img, img, jaccard_descriptor, [sample_comp_img]))

    descr_imgs_num = len(descr_imgs)
    if descr_imgs_num < 1:
        plt.imshow(mark_boundaries(descr_imgs[0], superpx_img)) 
    else:
        figs_num_hor = math.ceil(descr_imgs_num/2)
        if figs_num_hor == 1:
            figs_num_hor = 2
        figure_descr_imgs, ax = plt.subplots(2, figs_num_hor, figsize=(10, 10), sharex=True, sharey=True)
        for i_fig, fig in enumerate(descr_imgs):
            ax[i_fig%figs_num_hor, math.floor(i_fig/figs_num_hor)].imshow(mark_boundaries(descr_imgs[i_fig], superpx_img))

        for a in ax.ravel():
            a.set_axis_off()

    plt.tight_layout()
    plt.show()