# Inspired by https://gist.github.com/stefanv/05c57153a544bad6f3dae06c2c69abd2

from skimage import data, segmentation, measure, color, img_as_float
from skimage import io

import numpy as np
import matplotlib.pyplot as plt

from inspect import getsourcefile
import os
script_dir = os.path.dirname(getsourcefile(lambda:0))
os.chdir(script_dir)
os.chdir('../../../')


# image = img_as_float(data.chelsea())
image = io.imread('test_media/imgs/test_img_0.jpg')
# or use io.imread(filename) to read your own

# # A lot of online resources suggest LAB is the most suitable colour space:
# image_lab = color.rgb2lab(image)
# segments = segmentation.slic(image_lab, slic_zero=True)
# # Unfortunately, it seems RGB colour space (originally used) is better overall as it segments the ball well and the squares good enough
segments = segmentation.slic(image, slic_zero=True)

regions = measure.regionprops(segments)
colorfulness = np.zeros(image.shape[:2])

for region in regions:
    # Grab all the pixels from this region
    # Return an (N, 3) array
    coords = tuple(region.coords.T)
    values = image[coords]

    R, G, B = values.T

    # Calculate colorfulness
    rg = np.abs(R - G)
    yb = np.abs(0.5 * (R + G) - B)

    std_root = np.sqrt(np.std(rg) ** 2 + np.std(yb) ** 2)
    mean_root = np.sqrt(np.mean(rg) ** 2 + np.mean(yb) ** 2)

    colorfulness[coords] = std_root + (0.3 * mean_root)

hsv = color.rgb2hsv(image)
hsv[..., 2] *= colorfulness / colorfulness.max()

plt.imshow(colorfulness)
plt.show()

image_colourfulness_overlayed = color.hsv2rgb(hsv)
plt.figure()
plt.imshow(image_colourfulness_overlayed)

from skimage.segmentation import mark_boundaries
plt.figure()
plt.imshow(mark_boundaries(image_colourfulness_overlayed, segments))
plt.show()
