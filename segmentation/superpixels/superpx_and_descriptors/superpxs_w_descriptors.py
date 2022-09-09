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

    # Jaccard Similarity:
        # NOTE: Currently doesn't work too well due to padding pixels placed around the irregularly shaped region.
        # It's also VERY slow so not worth trying to implement it correctly.
    # sample_comp_img = skio.imread(imgs_dir+'sample_jacc_comp.png')
    # descr_imgs.append(gen_discriptor_img(superpx_img, img, jaccard_descriptor, [sample_comp_img]))

    # Dominant Colour:
    # descr_imgs.append(gen_discriptor_img(superpx_img, img, dominant_colour_descriptor))
    dom_colours_num = 3
    dom_colour_imgs = gen_discriptor_img(superpx_img, img, dominant_colour_descriptor, descr_dims=dom_colours_num*3, descr_func_args=[dom_colours_num])
    # dom_colour_imgs = gen_discriptor_img(superpx_img, img, dominant_colour_descriptor, descr_dims=dom_colours_num, descr_func_args=[dom_colours_num])
    # for i in range(int(dom_colour_imgs.shape[2]/dom_colours_num)):
    for i in range(dom_colours_num):
        descr_imgs.append(dom_colour_imgs[:, :, i:i+3])
        i += 3

    # Hue and Saturation average and MAD (mean absolute deviation)
    hue_mean_sat_mean_hue_MAD_sat_MAD = gen_discriptor_img(superpx_img, img, hs_stats_descriptor, descr_dims=4)
    descr_imgs.append(hue_mean_sat_mean_hue_MAD_sat_MAD[:,:,:3])
    descr_imgs.append(np.delete(hue_mean_sat_mean_hue_MAD_sat_MAD, 2, 2))

    # Avg. RGB:
    descr_imgs.append(gen_discriptor_img(superpx_img, img, avg_rgb_descriptor))

    return descr_imgs

def disp_descr_imgs(superpx_img, fig_and_ax, descr_imgs, descr_imgs_num):
    fig = fig_and_ax.pop(0)
    if descr_imgs_num < 1:
        fig.set_data(mark_boundaries(descr_imgs[0], superpx_img))
    else:
        figs_num_hor = math.ceil(descr_imgs_num/2)
        if figs_num_hor == 1:
            figs_num_hor = 2
        for i_fig, fig in enumerate(descr_imgs):
            # fig_and_ax[i_fig].set_data(mark_boundaries(descr_imgs[i_fig], superpx_img))
            asdf = descr_imgs[i_fig]
            asdfg = mark_boundaries(asdf, superpx_img)
            fig_and_ax[i_fig].set_data(asdfg)

        for a in ax.ravel():
            a.set_axis_off()

def grab_frame(capture):
    ret, frame = capture.read()
    return cv.cvtColor(frame, cv.COLOR_BGR2RGB)

use_video = False
do_loop = True
def on_close(event):
    global do_loop
    do_loop = False

if __name__ == '__main__':
    # Set the working directory to the root folder of the R&D git repo:
    script_dir = os.path.dirname(getsourcefile(lambda:0))
    os.chdir(script_dir)
    os.chdir('../../../')
    imgs_dir = 'test_media/imgs/'

    capture = None
    img = None
    if use_video:
        capture = cv.VideoCapture(-1)
        img = grab_frame(capture)
        plt.ion()
    else:
        img = skio.imread(imgs_dir+'test_img_0.jpg')

    plt.tight_layout()

    descr_imgs_num = 6

    # Display descriptor images:
        # Setup figures:
    descr_imgs_fig = None
    if descr_imgs_num < 1:
        descr_imgs_fig = [plt.figure()]
    else:
        figs_num_hor = math.ceil(descr_imgs_num/2)
        if figs_num_hor == 1:
            figs_num_hor = 2
        descr_imgs_fig, ax = plt.subplots(2, figs_num_hor, sharex=True, sharey=True)
        descr_imgs_fig = [descr_imgs_fig]
        for i_fig in range(descr_imgs_num):
            descr_imgs_fig.append(ax[i_fig%2, math.floor(0.5*i_fig)].imshow(img))
    descr_imgs_fig[0].canvas.mpl_connect('close_event', on_close)

    while(do_loop):
        if use_video:
            img = grab_frame(capture)

        superpx_img = create_superpx_img(img)
        descr_imgs = apply_superpx_descriptors(superpx_img, img)

        disp_descr_imgs(superpx_img, descr_imgs_fig, descr_imgs, len(descr_imgs))

        if not use_video:
            plt.show()
        else:
            plt.pause(0.005)

    if use_video:
        plt.ioff()
        plt.show()
        capture.release()
        
    # Destroy all the windows
    cv.destroyAllWindows()