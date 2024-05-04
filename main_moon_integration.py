import os
import numpy as np

from utils import read_fits_as_float, save_as_fits, get_filepaths_per_exptime, read_fits_header, extract_subheader
from parameters import MOON_DIR, MOON_STACKS_DIR


os.makedirs(MOON_STACKS_DIR, exist_ok=True)

# Make a dictionary that contains for each exposure time (key) a list of associated filepaths (value)
filepaths_per_exptime = get_filepaths_per_exptime(MOON_DIR)

# Need image shape to initialize stuff : we get it from the first filepath of the "first" key
header = read_fits_header(filepaths_per_exptime[list(filepaths_per_exptime)[0]][0]) 
shape = (header["NAXIS2"], header["NAXIS1"], header["NAXIS3"])

for exptime in filepaths_per_exptime.keys():
    print(f"Stacking {exptime}s exposures...")
    # Initialize stuff            
    stacked_img = np.zeros(shape)
    counts = 0
    # Loop over subs
    for filepath in filepaths_per_exptime[exptime]:
        # Read image
        img, header = read_fits_as_float(filepath)
        print("Adding image to the main stack...")
        stacked_img += img
        counts +=1

    stacked_img /= counts

    output_header = extract_subheader(header, ["EXPTIME", "PEDESTAL", "MOON-X", "MOON-Y"]) # common keywords
    save_as_fits(stacked_img, output_header, os.path.join(MOON_STACKS_DIR, f"{float(exptime):.5f}s.fits"))