import os
import numpy as np
from matplotlib import pyplot as plt

from utils import read_fits_as_float, crop, ht, combine_headers, save_as_fits, get_filepaths_per_exptime, read_fits_header, extract_subheader

from parameters import MOON_STACKS_DIR, SUN_STACKS_DIR, MERGED_STACKS_DIR
from parameters import SATURATION_VALUE, CLIP_EXP_TIME

from registration import prepare_img_for_detection
from scipy import ndimage

SIGMA = 2

os.makedirs(MERGED_STACKS_DIR, exist_ok=True)

sun_filepaths_per_exptime = get_filepaths_per_exptime(SUN_STACKS_DIR)
moon_filepaths_per_exptime = get_filepaths_per_exptime(MOON_STACKS_DIR)
exptimes = [float(exptime) for exptime in sun_filepaths_per_exptime.keys()]
exptimes.sort() # sort in ascending order

for i, exptime in enumerate(exptimes):
    moon_filepath, sun_filepath = moon_filepaths_per_exptime[str(exptime)][0], sun_filepaths_per_exptime[str(exptime)][0]
    img_moon, header_moon = read_fits_as_float(moon_filepath)
    img_sun, header_sun = read_fits_as_float(sun_filepath)

    print(f"Merging moon and sun stacks...")

    x_c = header_moon["MOON-X"]
    y_c = header_moon["MOON-Y"]

    if i == 0: # construct mask from the shortest exposure ; and re-use it for other exposures (for HDR consistency)
        # Extract binary moon mask
        threshold_mask = prepare_img_for_detection(img_moon, header_moon, CLIP_EXP_TIME, SATURATION_VALUE) < SATURATION_VALUE
        label_map, _ = ndimage.label(threshold_mask)
        moon_label = label_map[int(y_c), int(x_c)]
        moon_mask = (label_map == moon_label).astype('float')
        # Outward-only smoothing of the mask
        moon_mask = ndimage.gaussian_filter(moon_mask, sigma=SIGMA)
        moon_mask = np.clip(2*moon_mask, 0, 1)
        # We only want to add the moon, which correspond to dark pixels.
        # The border of the moon mask may be associated with bright pixels (esp. when sigma is high), which should be disregarded (or weighted less)
        # We correct the border of the mask by multiplying it with the inverse of the moon intensity (scaled to [0,1] in that region)
        border_pixels = (moon_mask > 0)*(moon_mask < 1)
        corrections = 1-img_moon.mean(axis=2)[border_pixels == 1]
        corrections = (corrections - corrections.min())/(corrections.max() - corrections.min())
        moon_mask[border_pixels == 1] *= corrections
    
    img_merged = moon_mask[:,:,None]*img_moon + (1-moon_mask)[:,:,None]*img_sun

    header_merged = combine_headers(header_moon, header_sun)
    save_as_fits(img_merged, header_merged, os.path.join(MERGED_STACKS_DIR, f"{float(exptime):.5f}s.fits"))
    #save_as_fits(moon_mask[:,:,None], None, os.path.join(MERGED_STACKS_DIR, f"moon_mask.fits"))

# fig, axes = plt.subplots(2,2)
# axes = axes.flatten()
# axes[0].imshow(ht(crop(img_moon, int(x_c), int(y_c)), m=0.1, low=0, high=0.13))
# axes[1].imshow(ht(crop(img_sun, int(x_c), int(y_c)), m=0.1, low=0, high=0.13))
# axes[2].imshow(ht(crop(img_merged, int(x_c), int(y_c)), m=0.1, low=0, high=0.13))
# axes[3].imshow(crop(moon_mask, int(x_c), int(y_c)), cmap='gray')
# plt.show()