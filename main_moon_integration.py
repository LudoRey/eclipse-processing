import os
import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter
from astropy.io.fits import Header

from utils import read_fits_as_float, save_as_fits, ht, crop, Timer
from parameters import MOON_RADIUS_DEGREE
from parameters import SATURATION_VALUE, CLIP_EXP_TIME
from parameters import IMAGE_SCALE, ROTATION
from parameters import INPUT_DIR, MOON_DIR, MOON_STACKS_DIR, SUN_DIR, FILENAME, REF_FILENAME


os.makedirs(MOON_STACKS_DIR, exist_ok=True)

img, header = read_fits_as_float(os.path.join(SUN_DIR, REF_FILENAME))
moon_x_c, moon_y_c = header["MOON-X"], header["MOON-Y"]
# Initialize stuff            
stacked_img = np.zeros(img.shape)
counts = 0

dirpath, _, filenames = next(os.walk(MOON_DIR))
for filename in filenames:
    #if filename in [REF_FILENAME, FILENAME]:
    if filename.endswith('.fits') and filename.startswith('0m25000s'):
        # Read image
        img, header = read_fits_as_float(os.path.join(MOON_DIR, filename))
        print("Adding image to the main stack...")
        stacked_img += img
        counts +=1

stacked_img /= counts

output_header = Header({"MOON-X" : moon_x_c, "MOON-Y" : moon_y_c})
exp_mm, exp_ss = f"{header["EXPTIME"]:.5f}".split('.')
save_as_fits(stacked_img, output_header, os.path.join(MOON_STACKS_DIR, f"{exp_mm}m{exp_ss}s.fits"))