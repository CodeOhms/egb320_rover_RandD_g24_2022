from inspect import getsourcefile
import os
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from skimage import io as skio
from skimage.segmentation import mark_boundaries

from superpx_transforms import *
from superpx_descriptors import *

def create_superpx_img(img):
    # Apply superpixel algos:
        # Watershed:
    # ws_markers, rgb_grad_im = watershed_get_markers(img)
    # return superpx_watershed_trans(rgb_grad_im, ws_markers)

        # SLIC:
    return superpx_slic_trans(img)

def apply_superpx_descriptors(superpx_img, img):
    descr_imgs = [ ]
    # Avg. RGB:
    descr_imgs.append(gen_discriptor_img(superpx_img, img, avg_rgb_descriptor))

    # Jaccard Similarity:
        # NOTE: Currently doesn't work too well due to padding pixels placed around the irregularly shaped region.
        # It's also VERY slow so not worth trying to implement it correctly.
    # sample_comp_img = skio.imread(imgs_dir+'sample_jacc_comp.png')
    # descr_imgs.append(gen_discriptor_img(superpx_img, img, jaccard_descriptor, [sample_comp_img]))

    # Dominant Colour:
    descr_imgs.append(gen_discriptor_img(superpx_img, img, dominant_colour_descriptor))

    # Hue and Saturation average and MAD (mean absolute deviation)
    hue_mean_sat_mean_hue_MAD_sat_MAD = gen_discriptor_img(superpx_img, img, hs_stats_descriptor, descr_dims=4)
    descr_imgs.append(hue_mean_sat_mean_hue_MAD_sat_MAD[:,:,:3])
    asdf = np.delete(hue_mean_sat_mean_hue_MAD_sat_MAD, 2, 2)
    descr_imgs.append(asdf)
    return descr_imgs

def disp_descr_imgs(superpx_img, fig_and_ax, descr_imgs, descr_imgs_num):
    if descr_imgs_num < 1:
        descr_imgs_fig = fig_and_ax[0]
        descr_imgs_fig.set_data(mark_boundaries(descr_imgs[0], superpx_img))
    else:
        figs_num_hor = math.ceil(descr_imgs_num/2)
        if figs_num_hor == 1:
            figs_num_hor = 2
        for i_fig, fig in enumerate(descr_imgs):
            fig_and_ax[i_fig].set_data(mark_boundaries(descr_imgs[i_fig], superpx_img))

        for a in ax.ravel():
            a.set_axis_off()

def grab_frame(capture):
    ret, frame = capture.read()
    return cv.cvtColor(frame, cv.COLOR_BGR2RGB)

use_video = True
def close(event):
    global use_video
    if event.key == 'c':
        plt.close(event.canvas.figure)
        use_video = False

if __name__ == '__main__':
    # Set the working directory to the root folder of the R&D git repo:
    script_dir = os.path.dirname(getsourcefile(lambda:0))
    os.chdir(script_dir)
    os.chdir('../../../')
    imgs_dir = 'test_media/imgs/'

    capture = None
    img = None
    if use_video:
        capture = cv.VideoCapture(0)
        img = grab_frame(capture)
        plt.ion()
    else:
        img = skio.imread(imgs_dir+'test_img_0.jpg')

    plt.tight_layout()

    descr_imgs_num = 4

    # Display descriptor images:
        # Setup figures:
    descr_imgs_fig = None
    if descr_imgs_num < 1:
        descr_imgs_fig = plt.figure()
    else:
        figs_num_hor = math.ceil(descr_imgs_num/2)
        if figs_num_hor == 1:
            figs_num_hor = 2
        _, ax = plt.subplots(2, figs_num_hor, sharex=True, sharey=True)
        descr_imgs_fig = [ ]
        for i_fig in range(descr_imgs_num):
            descr_imgs_fig.append(ax[i_fig%figs_num_hor, math.floor(i_fig/figs_num_hor)].imshow(img))

    while(True):
        if use_video:
            img = grab_frame(capture)

        superpx_img = create_superpx_img(img)
        descr_imgs = apply_superpx_descriptors(superpx_img, img)

        disp_descr_imgs(superpx_img, descr_imgs_fig, descr_imgs, len(descr_imgs))

        plt.pause(0.005)

        if not use_video:
            break

    plt.show()

    if use_video:
        plt.ioff()
        plt.show()
        capture.release()
        
    # Destroy all the windows
    cv.destroyAllWindows()