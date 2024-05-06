import os
import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter
from astropy.io.fits import Header
from sklearn.linear_model import LinearRegression

from utils import Timer, crop, ht, remove_pedestal, read_fits_as_float, save_as_fits, extract_subheader, get_filepaths_per_exptime, read_fits_header
from parameters import MOON_RADIUS_DEGREE
from parameters import SATURATION_VALUE, CLIP_EXP_TIME
from parameters import IMAGE_SCALE, ROTATION
from parameters import SUN_STACKS_DIR, MOON_STACKS_DIR

from disk import binary_disk
from hdr import saturation_weighting, equalize_brightness
from polar import angle_map

LOW_CLIPPING_THRESHOLD = 0.005
LOW_SMOOTHNESS = 0.001
HIGH_CLIPPING_THRESHOLD = 0.1 # if exptimes[i+1]/exptimes[i] = k, then we should have HIGH/LOW << k
HIGH_SMOOTHNESS = 0.01
BASE_EXP_TIME = 0.00025*4

SUN_HDR_DIR = "data\\totality\\sun_hdr"
os.makedirs(SUN_HDR_DIR, exist_ok=True)

moon_radius_pixels = MOON_RADIUS_DEGREE * 3600 / IMAGE_SCALE
EXTRA_RADIUS_PIXELS = 10
moon_radius_pixels += EXTRA_RADIUS_PIXELS

filepaths_per_exptime = get_filepaths_per_exptime(SUN_STACKS_DIR)
moon_filepaths_per_exptime = get_filepaths_per_exptime(MOON_STACKS_DIR)
exptimes = [float(exptime) for exptime in filepaths_per_exptime.keys()]
exptimes.sort() # sort in ascending order

#exptimes = exptimes[:2]

# Initialize stuff
header = read_fits_header(filepaths_per_exptime[str(exptimes[0])][0])
moon_header = read_fits_header(moon_filepaths_per_exptime[str(exptimes[0])][0])
shape = (header["NAXIS2"], header["NAXIS1"], header["NAXIS3"])
moon_mask = binary_disk(moon_header["MOON-X"], moon_header["MOON-Y"], radius=moon_radius_pixels, shape=shape[0:2])
img_theta = angle_map(header["SUN-X"], header["SUN-Y"], shape=shape[0:2])

# Read image to fit (and remove pedestal), extract mask and rescale image to obtain scaled irradiance
img_y, header_y = read_fits_as_float(filepaths_per_exptime[str(exptimes[-1])][0])
# Compute mask and weights
mask_y = (img_y.max(axis=2) > LOW_CLIPPING_THRESHOLD) * (img_y.max(axis=2) < HIGH_CLIPPING_THRESHOLD)
weights = saturation_weighting(img_y.max(axis=2), 0, HIGH_CLIPPING_THRESHOLD, LOW_SMOOTHNESS, HIGH_SMOOTHNESS)
save_as_fits(weights, None, os.path.join(SUN_HDR_DIR, f"weights_{exptimes[-1]:.5f}.fits"))
weights *= header_y["EXPTIME"]**2
# Compute scaled irradiance
img_y = remove_pedestal(img_y, header_y) * BASE_EXP_TIME / header_y["EXPTIME"]

# Add weighted scaled irradiance to the HDR image
hdr_img = weights[:,:,None] * img_y
sum_weights = weights

for i in reversed(range(len(exptimes)-1)):
    # Read image to fit (and remove pedestal), extract mask and rescale image
    img_x, header_x = read_fits_as_float(filepaths_per_exptime[str(exptimes[i])][0])
    # Compute mask and weights
    mask_x = (img_x.max(axis=2) > LOW_CLIPPING_THRESHOLD) * (img_x.max(axis=2) < HIGH_CLIPPING_THRESHOLD)
    if i == 0:
        weights = saturation_weighting(img_x.max(axis=2), LOW_CLIPPING_THRESHOLD, 1, LOW_SMOOTHNESS, HIGH_SMOOTHNESS)
    else:
        weights = saturation_weighting(img_x.max(axis=2), LOW_CLIPPING_THRESHOLD, HIGH_CLIPPING_THRESHOLD, LOW_SMOOTHNESS, HIGH_SMOOTHNESS)
    save_as_fits(weights, None, os.path.join(SUN_HDR_DIR, f"weights_{exptimes[i]:.5f}.fits"))
    weights *= header_x["EXPTIME"]**2
    # Compute scaled irradiance
    img_x = remove_pedestal(img_x, header_x) * BASE_EXP_TIME / header_x["EXPTIME"]

    # Create combined mask 
    mask = mask_x * mask_y * ~moon_mask
    
    # Fit image
    print("Equalizing the brightness...")
    img_x = equalize_brightness(img_x, img_theta, img_y, mask, return_coeffs=False)

    # Add weighted scaled irradiance to the HDR image
    hdr_img += weights[:,:,None] * img_x
    sum_weights += weights

    # Fitted image becomes the new reference
    img_y = img_x
    mask_y = mask_x

hdr_img /= sum_weights[:,:,None]
#hdr_img = np.maximum(hdr_img/hdr_img.max(), 0) # rescale to [0,1]
print(hdr_img.max(), hdr_img.min())
hdr_img = np.clip(hdr_img, 0, 1)

output_header = extract_subheader(header_x, ["SUN-X", "SUN-Y"])

save_as_fits(hdr_img, output_header, os.path.join(SUN_HDR_DIR, "hdr.fits"), convert_to_uint16=False)

plt.imshow(ht(crop(hdr_img, int(header_x["SUN-X"]), int(header_x["SUN-Y"]), 1024, 1024), m=0.1))
plt.show()