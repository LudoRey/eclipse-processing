import os
import numpy as np
from astropy.coordinates import EarthLocation
from astropy.time import Time

from .lib.registration import moon_detection
from .lib.fits import read_fits_as_float, update_fits_header

def main(input_dir, image_scale):
    # Upper bound on the apparent moon radius (coarse estimate used for Canny threshold)
    moon_radius_degree = 0.2783
    moon_radius_pixels = moon_radius_degree * 3600 / image_scale

    dirpath, _, filenames = next(os.walk(input_dir)) # not going into subfolders
    for filename in filenames:
        #if filename == "0.25000s_2024-04-09_02h42m31s.fits" or filename == "0.25000s_2024-04-09_02h40m33s.fits":

        img, header = read_fits_as_float(os.path.join(dirpath, filename))
        moon_x, moon_y, moon_radius = moon_detection(img, moon_radius_pixels)
        
        # TODO : add callbacks ?
        update_fits_header(os.path.join(dirpath, filename),
                            {'MOON-X': (moon_x, 'X-coordinate of the moon center.'),
                            'MOON-Y': (moon_y, 'Y-coordinate of the moon center.'),
                            'MOON-R': (moon_radius, 'Radius of the moon.')})

if __name__ == "__main__":

    from parameters import IMAGE_SCALE
    from parameters import MEASURED_TIME, UTC_TIME, LATITUDE, LONGITUDE
    from parameters import INPUT_DIR

    REF_FILENAME = "0.01667s_2024-04-09_02h42m25s.fits"

    main(INPUT_DIR, IMAGE_SCALE)