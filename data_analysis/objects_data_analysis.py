import glob
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
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
        hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
        h = hsv[:,:,0]
        h = h.reshape(h.shape[0]*h.shape[1])
        s = hsv[:,:,1]
        s = s.reshape(s.shape[0]*s.shape[1])
        hue_packed = np.append(hue_packed, h)
        sat_packed = np.append(sat_packed, s)

        np.append(hue_MAD_packed, MAD(h))
        np.append(sat_MAD_packed, MAD(s))

    return (hue_packed, sat_packed, hue_MAD_packed, sat_MAD_packed)

def density_plot_generate(data_set, smoothing=.5):
    densities = [ ]
    for data in data_set:
        density = stats.gaussian_kde(data)
        density.covariance_factor = lambda : smoothing
        density._compute_covariance()
        densities.append(density)
    
    return densities

def density_plot_display(density_data, domain=np.linspace(0,20,200), figure=plt.figure()):
    (hs_density, h_MAD_s_MAD_density) = density_data
    figure[0].set_data(domain, hs_density(domain))
    figure[1].set_data(domain, h_MAD_s_MAD_density(domain))


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
    gb_figure = plt.figure()
    _, gb_subplts_ax = plt.subplots(1, 2, sharex=True, sharey=True)

    # rock_figure = plt.figure()
    # _, rock_subplts_ax = plt.subplots(1, 2, sharex=True, sharey=True)
    
    # obstacle_figure = plt.figure()
    # _, obstacle_subplts_ax = plt.subplots(1, 2, sharex=True, sharey=True)

    # Display density data:
    density_plot_display(gb_data_density, figure=gb_subplts_ax)
    # density_plot_display(rock_data_density, figure=rock_subplts_ax)
    # density_plot_display(obstacle_data_density, figure=obstacle_subplts_ax)
    
