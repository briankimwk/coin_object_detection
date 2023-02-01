# This is a practice Python script for scikit image processing

from skimage import data
from skimage.exposure import histogram
import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage as ndi
from skimage.filters import sobel
from skimage.segmentation import watershed

coins = data.coins()

# check the pixel values to identify likely edge values
hist, hist_centers = histogram(coins)
# plt.plot(hist)
# plt.show()
# majority of pixels are between 30 and 150 (the coins)

markers = np.zeros_like(coins)
markers[coins < 30] = 1
markers[coins > 150] = 2

elevation_map = sobel(coins)

segmentation = watershed(elevation_map, markers)
segmentation = ndi.binary_fill_holes(segmentation - 1)

# create subplot grid for before and after images
f, axarr = plt.subplots(1, 2)


# use the created array to output images
axarr[0].imshow(coins)
axarr[1].imshow(segmentation, cmap="gray")
plt.show()
