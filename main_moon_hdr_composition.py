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

from hdr import saturation_weighting

LOW_CLIPPING_THRESHOLD = 0.005
LOW_SMOOTHNESS = 0.001
HIGH_CLIPPING_THRESHOLD = 0.1 # if exptimes[i+1]/exptimes[i] = k, then we should have HIGH/LOW << k
HIGH_SMOOTHNESS = 0.01
BASE_EXP_TIME = 0.00025*4

MOON_HDR_DIR = "data\\totality\\moon_hdr"
os.makedirs(MOON_HDR_DIR, exist_ok=True)

filepaths_per_exptime = get_filepaths_per_exptime(MOON_STACKS_DIR)
exptimes = [float(exptime) for exptime in filepaths_per_exptime.keys()]
exptimes.sort() # sort in ascending order

# Initialize stuff
header = read_fits_header(filepaths_per_exptime[str(exptimes[0])][0])
shape = (header["NAXIS2"], header["NAXIS1"], header["NAXIS3"])
hdr_img = np.zeros(shape)
sum_weights = np.zeros(shape[0:2])

for i in range(len(exptimes)):
    # Read image
    img, header = read_fits_as_float(filepaths_per_exptime[str(exptimes[i])][0])
    # Compute weight
    if i == 0:
        weights = saturation_weighting(img.max(axis=2), LOW_CLIPPING_THRESHOLD, 1, LOW_SMOOTHNESS, HIGH_SMOOTHNESS)
    elif i == len(exptimes) - 1:
        weights = saturation_weighting(img.max(axis=2), 0, HIGH_CLIPPING_THRESHOLD, LOW_SMOOTHNESS, HIGH_SMOOTHNESS)
    else:
        weights = saturation_weighting(img.max(axis=2), LOW_CLIPPING_THRESHOLD, HIGH_CLIPPING_THRESHOLD, LOW_SMOOTHNESS, HIGH_SMOOTHNESS)
    save_as_fits(weights, None, os.path.join(MOON_HDR_DIR, f"weights_{exptimes[i]:.5f}.fits"))
    weights *= header["EXPTIME"]**2
    # Compute scaled irradiance
    img = remove_pedestal(img, header) * BASE_EXP_TIME / header["EXPTIME"]

    # Add weighted scaled irradiance to the HDR image
    hdr_img += weights[:,:,None] * img
    sum_weights += weights

hdr_img /= sum_weights[:,:,None]
#hdr_img = np.maximum(hdr_img/hdr_img.max(), 0) # rescale to [0,1]
print(hdr_img.max(), hdr_img.min())
hdr_img = np.clip(hdr_img, 0, 1)

output_header = extract_subheader(header, ["MOON-X", "MOON-Y"])

save_as_fits(hdr_img, output_header, os.path.join(MOON_HDR_DIR, "hdr.fits"), convert_to_uint16=False)

plt.imshow(ht(crop(hdr_img, int(header["MOON-X"]), int(header["MOON-Y"]), 1024, 1024), m=0.1))
plt.show()