from inspect import getsourcefile
from locale import normalize
import os
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
# from matplotlib.animation import FuncAnimation
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
    dom_colours_num = 2
    dom_colour_imgs = gen_discriptor_img(superpx_img, img, dominant_colour_descriptor, descr_dims=dom_colours_num*3, descr_func_args=[dom_colours_num])
    for i in range(dom_colours_num):
        descr_imgs.append(show_boundaries(dom_colour_imgs[:, :, i:i+3], superpx_img, output_channels=slice(0,3), output_dtype=np.uint8))
        i += 3

    # Hue and Saturation average and MAD (mean absolute deviation)
    hue_mean_sat_mean_hue_MAD_sat_MAD = gen_discriptor_img(superpx_img, img, hs_stats_descriptor, img_dtype=np.float32, descr_dims=4)
    for i in range(hue_mean_sat_mean_hue_MAD_sat_MAD.shape[2]):
        hsv_descr_img = hue_mean_sat_mean_hue_MAD_sat_MAD[:,:,i]
        descr_imgs.append(show_boundaries(hsv_descr_img, superpx_img, output_scale=1, output_dtype=np.float32))

    # Hue MAD from given hue:
    hue_mads = [2, 83, 207] # Sample, obstacle, rock
    hue_mads_imgs = [gen_discriptor_img(superpx_img, img, MAD_from_hue, descr_func_args=[hue_mads[i]], descr_dims=1, img_dtype=np.float32)
    for i in range(3)
    ]
    for i in range(len(hue_mads)):
        descr_imgs.append(show_boundaries(hue_mads_imgs[i][:,:,0], superpx_img, output_scale=1, output_dtype=np.float32))

    # Avg. RGB:
    descr_imgs.append(
        show_boundaries(gen_discriptor_img(superpx_img, img, avg_rgb_descriptor),
        superpx_img, output_channels=slice(0,3), output_dtype=np.uint8)
    )

    return descr_imgs

def disp_descr_imgs(titles, fig_and_ax, descr_imgs, colorbar_scales, descr_imgs_num):
    if descr_imgs_num < 1:
        fig_and_ax[0].set_data(descr_imgs[0])
    else:
        figs_num_hor = math.ceil(descr_imgs_num/2)
        if figs_num_hor == 1:
            figs_num_hor = 2
        i_cbar = 0
        for i_ax, axis in enumerate(fig_and_ax[1:]):
            axis.set_title(titles[i_ax])
            descr_image = descr_imgs[i_ax]
            img_max = None
            if len(descr_image.shape) == 2:
                img_max = descr_image.max()
                cbar_scale = colorbar_scales[i_cbar]
                i_cbar += 1
                if img_max <= 2/3*cbar_scale:
                    img_max *= 1.5
                else:
                    img_max = cbar_scale
                nm = plt.Normalize(vmin=0, vmax=img_max)
            descr_img_dtype = descr_image.dtype
            axes_img = axis.get_images()[0]
            if img_max is not None:
                axes_img.set_norm(nm)
            axes_img.set_data(descr_image)
            if len(descr_image.shape) == 2:
                descr_imgs_fig[0].colorbar(axes_img, ax=axis)
            plt.draw()

        for a in ax.ravel():
            a.set_axis_off()
        
        plt.show()

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


    # Display descriptor images:
        # Setup figures:
    descr_imgs_num = 10
    colour_maps = ['rgb', 'rgb', 'cividis', 'cividis', 'cividis', 'cividis', 'cividis', 'cividis', 'cividis', 'rgb']
    descr_imgs_fig = None
    colour_bars = [None for i in range(descr_imgs_num)]
    if descr_imgs_num < 1:
        descr_imgs_fig = [plt.figure()]
    else:
        figs_num_hor = math.ceil(descr_imgs_num/2)
        if figs_num_hor == 1:
            figs_num_hor = 2
        descr_imgs_fig, ax = plt.subplots(2, figs_num_hor, sharex=True, sharey=True)
        descr_imgs_fig = [descr_imgs_fig]
        for i_ax in range(descr_imgs_num):
            axis = ax[i_ax%2, math.floor(0.5*i_ax)]
            colour_m = colour_maps[i_ax]
            descr_imgs_fig.append(axis)
            if colour_m == 'rgb':
                axis.imshow(img)
            else:
                axis.imshow(np.zeros(img.shape[:2]), cmap=colour_m)
                
    descr_imgs_fig[0].canvas.mpl_connect('close_event', on_close)

    while(True):
        superpx_img = create_superpx_img(img)
        descr_imgs = apply_superpx_descriptors(superpx_img, img)
        subpl_titles = ['1st Dominant Colour', '2nd Dominant Colour', 'Hue Avg.', 'Hue MAD', 'Saturation Avg.', 'Saturation MAD', 'MAD from hue of Samples', 'MAD from hue of Obstacles', 'MAD from hue of Rocks', 'Average RGB']
        cbar_scales = [360, 1, 360, 1, 180, 180, 180]
        disp_descr_imgs(subpl_titles, descr_imgs_fig, descr_imgs, cbar_scales, len(descr_imgs))

        if use_video:
            plt.pause(0.005)

            if not do_loop:
                plt.ioff()
                capture.release()
                break
        else:
            break
    
    plt.show()
        
    # Destroy all the windows
    cv.destroyAllWindows()