import os
import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter
from astropy.io.fits import Header

from utils import read_fits_as_float, save_as_fits, ht, crop, Timer, extract_subheader, read_fits_header, get_filepaths_per_exptime
from parameters import MOON_RADIUS_DEGREE
from parameters import SATURATION_VALUE, CLIP_EXP_TIME
from parameters import IMAGE_SCALE, ROTATION
from parameters import INPUT_DIR, MOON_DIR, SUN_STACKS_DIR, SUN_DIR, FILENAME, REF_FILENAME

def binary_disk(x_c: float, y_c: float, radius: float, shape: np.array):
    # Not used
    x, y = np.arange(shape[1]), np.arange(shape[0])
    X, Y = np.meshgrid(x, y)
    binary_disk = np.sqrt((X-x_c)**2 + (Y-y_c)**2) <= radius
    return binary_disk 

def linear_falloff_disk(x_c: float, y_c: float, radius: float, shape: np.array, smoothness: float = 10, return_dist: bool = False):
    x, y = np.arange(shape[1]), np.arange(shape[0])
    X, Y = np.meshgrid(x, y)
    R = np.sqrt((X-x_c)**2 + (Y-y_c)**2)
    smooth_disk = np.clip((radius-R+smoothness)/smoothness, 0, 1)
    if return_dist:
        return smooth_disk, R
    else:
        return smooth_disk

moon_radius_pixels = MOON_RADIUS_DEGREE * 3600 / IMAGE_SCALE
EXTRA_RADIUS_PIXELS = 2
moon_radius_pixels += EXTRA_RADIUS_PIXELS
SMOOTHNESS = 10

os.makedirs(SUN_STACKS_DIR, exist_ok=True)

# Make a dictionary that contains for each exposure time (key) a list of associated filepaths (value)
filepaths_per_exptime = get_filepaths_per_exptime(SUN_DIR)

# Need image shape to initialize stuff : we get it from the first filepath of the "first" key
header = read_fits_header(filepaths_per_exptime[list(filepaths_per_exptime)[0]][0]) 
shape = (header["NAXIS2"], header["NAXIS1"], header["NAXIS3"])

for exptime in filepaths_per_exptime.keys():
    print(f"Stacking {exptime}s exposures...")
    # Initialize stuff
    stacked_img = np.zeros(shape)
    filler_img = np.zeros(shape)
    sum_weights = np.zeros(shape[0:2])
    max_dist_to_moon_center = np.zeros(shape[0:2])
    # Loop over subs
    for filepath in filepaths_per_exptime[exptime]:
        # Read image
        img, header = read_fits_as_float(filepath)
        # Get the moon center coordinates of the current frame
        moon_x_c, moon_y_c = header["MOON-X"], header["MOON-Y"] 

        ### Update main stack image 
        # 1. Add pixels outside the moon mask (dist_to_moon_center is useful later)
        print("Computing moon mask used to weight image...")
        moon_mask, dist_to_moon_center = linear_falloff_disk(moon_x_c, moon_y_c, moon_radius_pixels, shape[0:2], smoothness=SMOOTHNESS, return_dist=True)
        print("Adding weighted image to the main stack...")
        stacked_img += img * (1-moon_mask[:,:,None])
        # 2. Update sum_weights
        new_in_stack = np.logical_and(moon_mask < 1, sum_weights == 0) # used later to remove those pixels from the filler image
        sum_weights += 1-moon_mask

        ### Update filler image (which only contains pixels NOT in the main stack, i.e with sum_weights == 0)
        print("Updating filler image...")
        # 1. Remove pixels that are now in the main stack
        filler_img[new_in_stack] = 0
        # 2. For pixels that are not in the main stack : use those that are the farthest so far from the moon center
        # Effectively, only the subs nearest to C2 and C3 will be used.
        farthest_from_moon_center_so_far = dist_to_moon_center >= max_dist_to_moon_center
        mask = (sum_weights == 0) * farthest_from_moon_center_so_far
        filler_img[mask] = img[mask]
        #print(f"Updated {np.count_nonzero(mask)/np.count_nonzero(sum_weights == 0)*100:.0f}% pixels of the filler image.")
        # 3. Update max distance tracker
        max_dist_to_moon_center[farthest_from_moon_center_so_far] = dist_to_moon_center[farthest_from_moon_center_so_far]

    stacked_img[sum_weights != 0] /= sum_weights[sum_weights != 0, None]
    merged_img = np.copy(stacked_img)
    merged_img[sum_weights == 0] = filler_img[sum_weights == 0]

    output_header = extract_subheader(header, ["EXPTIME", "PEDESTAL", "SUN-X", "SUN-Y"]) # common keywords
    exp_mm, exp_ss = f"{header["EXPTIME"]:.5f}".split('.')
    save_as_fits(merged_img, output_header, os.path.join(SUN_STACKS_DIR, f"{exp_mm}m{exp_ss}s.fits"))

# stacked_img = crop(stacked_img, int(sun_x_c), int(sun_y_c))
# filler_img = crop(filler_img, int(sun_x_c), int(sun_y_c))
# merged_img = crop(merged_img, int(sun_x_c), int(sun_y_c))

# low, high = merged_img.min(), merged_img.max()
# stacked_img = ht(stacked_img, m=0.1, low=low, high=high)
# filler_img = ht(filler_img, m=0.1, low=low, high=high)
# merged_img = ht(merged_img, m=0.1, low=low, high=high)

# sum_weights = crop(sum_weights, int(sun_x_c), int(sun_y_c))
# mask = crop(mask, int(sun_x_c), int(sun_y_c))

# fig, axes = plt.subplots(2,2)
# axes = axes.flatten()
# axes[0].imshow(stacked_img)
# axes[1].imshow(filler_img)
# axes[2].imshow(merged_img)
# axes[3].imshow(sum_weights)

# plt.figure()
# plt.imshow(merged_img)

# # plt.figure()
# # plt.imshow(filler_img_contributors)

# plt.show()