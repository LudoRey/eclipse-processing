if __name__ == "__main__":
    from _import import add_project_root_to_path
    add_project_root_to_path()

import os
import numpy as np

from core.lib import registration, fits

def main(input_dir,
         image_scale,
         moon_radius_degree=0.278,
         num_clipped_multiplier=1.3,
         num_edge_multiplier=1.0,
         smoothing=None,
         *,
         img_callback=lambda img: None,
         checkstate=lambda: None):
    '''
    Parameters
    ----------
    - input_dir
    - image_scale: resolution in arcsec/pixels.
    - moon_radius_degree: apparent moon radius in degrees. Default corresponds to the perigee, i.e. an upper bound.
    Together with `image_scale`, determines `moon_radius_pixels`.
    - num_clipped_multiplier: determines the number of clipped pixels (especially used for shorter subs).
    Formula: `pi*(num_clipped_multiplier**2 - 1)*moon_radius_pixels**2`. Corresponds to the outer radius of an annulus, given in moon radii.
    Should be large enough so that a complete annulus is clipped, but not too large so
    - num_edge_multiplier: determines the number of retained edge pixels. Formula: `num_edge_multiplier*2*np.pi*moon_radius_pixels`.
    Default is 1, which corresponds to the moon circonference.
    '''
    # Default moon radius is an upper bound (coarse estimate used for threshold)
    moon_radius_pixels = np.rint(moon_radius_degree * 3600 / image_scale)
    # Because of brightness variations, the multiplier should be large enough so that a complete annulus is clipped
    num_clipped_pixels = np.rint(np.pi*(num_clipped_multiplier**2 - 1)*moon_radius_pixels**2)
    # The number of pixels that correspond to the edge of the moon (given here by its circonference times a multiplier)
    num_edge_pixels = np.rint(num_edge_multiplier*2*np.pi*moon_radius_pixels)
    # Note : we are losing precision by rounding to nearest integer at each step to be consistent with the GUI
    # This does not really matter here, we do not need to be precise

    dirpath, _, filenames = next(os.walk(input_dir)) # not going into subfolders
    for filename in filenames:
        #if filename == "0.25000s_2024-04-09_02h42m31s.fits" or filename == "0.25000s_2024-04-09_02h40m33s.fits":

        img, header = fits.read_fits_as_float(os.path.join(dirpath, filename))
        img = registration.moon.preprocess(img, num_clipped_pixels, smoothing)
        checkstate()
        img_callback(img)

        moon_x, moon_y, moon_radius, img = registration.moon.detect(img, num_edge_pixels)
        checkstate()
        img_callback(img)
        
        fits.update_fits_header(os.path.join(dirpath, filename),
                                {'MOON-X': (moon_x, 'X-coordinate of the moon center.'),
                                'MOON-Y': (moon_y, 'Y-coordinate of the moon center.'),
                                'MOON-R': (moon_radius, 'Radius of the moon.')})
        checkstate()

if __name__ == "__main__":
    import sys
    from core.lib.utils import ColorTerminalStream
    sys.stdout = ColorTerminalStream()

    from core.parameters import IMAGE_SCALE
    from core.parameters import MEASURED_TIME, UTC_TIME, LATITUDE, LONGITUDE
    from core.parameters import INPUT_DIR

    REF_FILENAME = "0.01667s_2024-04-09_02h42m25s.fits"

    main(INPUT_DIR, IMAGE_SCALE)