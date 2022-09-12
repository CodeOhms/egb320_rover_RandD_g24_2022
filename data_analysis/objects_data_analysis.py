import glob
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import seaborn as sns
# import scipy as sp
from scipy import stats

def open_images(relative_dir, file_pattern='*.png'):
    files = glob.glob(relative_dir+'/'+file_pattern)
    files.sort()

    images = [ ]

    for f in files:
        images.append(cv.imread(f))

    return images

def MAD(data):
    """
    Mean Absolute Deviation.
    """

    return np.mean(np.absolute(data - np.mean(data)))

def prepare_img_data(image_list):
    hue_packed = np.array([])
    sat_packed = np.array([])
    hue_MAD_packed = np.array([])
    sat_MAD_packed = np.array([])

    for image in image_list:
        image32 = np.float32(image)
        hsv = cv.cvtColor(image32, cv.COLOR_BGR2HSV_FULL)
        h = hsv[:,:,0]
        h = h.reshape(h.shape[0]*h.shape[1])
        s = hsv[:,:,1]
        s = s.reshape(s.shape[0]*s.shape[1])
        hue_packed = np.append(hue_packed, h)
        sat_packed = np.append(sat_packed, s)

        hue_MAD_packed = np.append(hue_MAD_packed, MAD(h))
        sat_MAD_packed = np.append(sat_MAD_packed, MAD(s))

    return (hue_packed, sat_packed, hue_MAD_packed, sat_MAD_packed)

def density_plot_generate(data_set, smoothing=.5):
    densities = [ ]
    for data in data_set:
        density = stats.gaussian_kde(data)
        density.covariance_factor = lambda : smoothing
        density._compute_covariance()
        densities.append(density)

        # densities.append(data)
    
    return densities

def density_plot_display(densities, domain, axs, titles):
    for ax, density, title in zip(axs, densities, titles):
        ax.plot(domain, density(domain))
        # sns.kdeplot(density)
        ax.set_xlabel(title)
        ax.set_ylabel('Density')

if __name__ == "__main__":
    # Read all the sample images:
    gb_images = open_images('test_media/imgs/objects_data_analysis/golf_ball_samples_imgs_da/gb_da_edited', '*.png')
    # rock_images = open_images('test_media/imgs/objects_data_analysis/rock_imgs_da', '.png')
    # obstacle_images = open_images('test_media/imgs/objects_data_analysis/obstacle_imgs_da', '.png')

    # Prepare image data:
    gb_img_data = prepare_img_data(gb_images)
    # rock_img_data = prepare_img_data(rock_images)
    # obstacle_img_data = prepare_img_data(obstacle_images)

    # Calculate densities of image data:
        # Golf ball sample:
    gb_data_densities = density_plot_generate(gb_img_data)
    
    #     # Rock:
    # rock_data_density = density_plot_generate(rock_img_data)

    #     # Obstacle:
    # obstacle_data_density = density_plot_generate(obstacle_img_data)
    
    # Prepare figures and plots:
    gb_figure, gb_subplts_axs = plt.subplots(4, 1)
    gb_figure.tight_layout()
    gb_figure.suptitle('Orange Golf Ball Samples HSV Analysis')

    # rock_figure = plt.figure()
    # _, rock_subplts_ax = plt.subplots(1, 2, sharex=True, sharey=True)
    
    # obstacle_figure = plt.figure()
    # _, obstacle_subplts_ax = plt.subplots(1, 2, sharex=True, sharey=True)

    # Display density data:
    dens_plt_titles = ('Hue', 'Saturation', 'Hue Mean Absolute Deviation (MAD)', 'Sat MAD')
        # Hue and hue MAD:
    density_plot_display(gb_data_densities[0::2], np.linspace(0,360,361), gb_subplts_axs[0::2], dens_plt_titles[0::2])
        # Sat and sat MAD:
    density_plot_display(gb_data_densities[1::2], np.linspace(0,1,100), gb_subplts_axs[1::2], dens_plt_titles[1::2])
    # density_plot_display(rock_data_density, figure=rock_subplts_ax)
    # density_plot_display(obstacle_data_density, figure=obstacle_subplts_ax)
    
    plt.show()
