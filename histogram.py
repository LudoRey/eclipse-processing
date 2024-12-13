import os
from matplotlib import pyplot as plt
import numpy as np
from astropy.coordinates import EarthLocation
from astropy.time import Time
import warnings

from lib.registration import translate, moon_detection, get_moon_radius, moon_preprocessing
from lib.fits import read_fits_as_float, save_as_fits
from lib.display import auto_ht_params, ht_lut, compute_statistics, center_crop
from lib.polar import warp_cart_to_polar


def find_local_minima(values, neighborhood_radius=50):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning) # ignore all NaNs warning in np.nanmin

        minima_indices = []

        for i in range(len(values)):
            left_bound = max(0, i - neighborhood_radius)
            right_bound = min(len(values), i + neighborhood_radius + 1)

            if values[i] == np.nanmin(values[left_bound:right_bound]):
                minima_indices.append(i)

    return minima_indices
        
def main(input_dir,
         latitude,
         longitude,
         time_offset,
         image_scale,
         ref_filename,
         other_filename):

    fig, ax = plt.subplots()

    # Load reference image
    img, header = read_fits_as_float(os.path.join(input_dir, ref_filename))
    # Compute clipping value
    for i, color in enumerate(['red', 'green', 'blue']):
        hist, bin_edges = np.histogram(img[:,:,i], bins=1000, range=[0,1])

        

        # Replace zeros by NaNs
        values = hist.astype(np.float32)
        values[hist == 0] = np.nan
        # Find local minima
        local_minima = find_local_minima(values, 5)
        print(local_minima)
        # 
        first_bin_index = np.searchsorted(bin_edges, img[:,:,i].min(), side='right') - 1 
        last_bin_index = np.searchsorted(bin_edges, img[:,:,i].max(), side='right') - 1
        print(first_bin_index, last_bin_index)

        ax.plot(values, color=color)

    ax.set_yscale('log')
    
    plt.show()

if __name__ == "__main__":

    from parameters import IMAGE_SCALE
    from parameters import TIME_OFFSET, LATITUDE, LONGITUDE
    from parameters import INPUT_DIR

    ref_filename = "0.25000s_2024-04-09_02h40m33s.fits"
    #ref_filename = "0.00025s_2024-04-09_02h43m02s.fits"
    other_filename = "0.25000s_2024-04-09_02h42m31s.fits"

    main(INPUT_DIR,
         LATITUDE,
         LONGITUDE,
         TIME_OFFSET,
         IMAGE_SCALE,
         ref_filename,
         other_filename)