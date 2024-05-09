import numpy as np
import matplotlib.pyplot as plt

from utils import crop_inset, read_fits_as_float, Timer
from polar import angle_map, radius_map
from achf import achf_kernel_at_ij, new_achf_kernel_at_ij

IMAGE_FILEPATH = "data\\totality\\merged_hdr\\hdr.fits"
MOON_MASK_FILEPATH = "data\\totality\\merged_hdr\\moon_mask.fits"

SIGMA = 1

img, header = read_fits_as_float(IMAGE_FILEPATH)
mask, _ = read_fits_as_float(MOON_MASK_FILEPATH)
x_c, y_c = header["SUN-X"], header["SUN-Y"]
img_gray = img.mean(axis=2) 
shape = img_gray.shape

theta, rho = angle_map(x_c, y_c, shape), radius_map(x_c, y_c, shape)

kernel = new_achf_kernel_at_ij(1000, 1000, theta, rho, SIGMA)

fig, axes = plt.subplots(1,3)

axes[0].imshow(theta); axes[0].set_title('Angle')
axes[1].imshow(rho); axes[1].set_title('Radius')
axes[2].imshow(kernel); axes[2].set_title('Kernel')

plt.show()