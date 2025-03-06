if __name__ == "__main__":
    from _import import add_project_root_to_path
    add_project_root_to_path()

import os
import numpy as np
from datetime import datetime
import scipy.interpolate

from core.lib import registration, fits, interpolation, transform
from core.lib.utils import cprint, Timer

def main(input_dir,
         ref_filename,
         anchor_filenames,
         moon_registered_dir,
         sun_registered_dir,
         image_scale,
         moon_radius_degree=0.278,
         num_clipped_multiplier=1.3,
         num_edge_multiplier=1.0,
         smoothing=None,
         blend_factor=0.75,
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
    - num_edge_multiplier: determines the number of proposed edge pixels. Formula: `num_edge_multiplier*2*np.pi*moon_radius_pixels`.
    Default is 1, which corresponds to the moon circonference. The final number of edge pixels is lower than that due to non-maximum suppression.
    '''
    # Default moon radius is an upper bound (coarse estimate used for threshold)
    moon_radius_pixels = np.rint(moon_radius_degree * 3600 / image_scale)
    # Because of brightness variations, the multiplier should be large enough so that a complete annulus is clipped
    num_clipped_pixels = np.rint(np.pi*(num_clipped_multiplier**2 - 1)*moon_radius_pixels**2)
    # The number of pixels that correspond to the edge of the moon (given here by its circonference times a multiplier)
    num_edge_pixels = np.rint(num_edge_multiplier*2*np.pi*moon_radius_pixels)
    # Note : we are losing precision by rounding to nearest integer at each step to be consistent with the GUI
    # This does not really matter here, we do not need to be precise

    # Initialize trackers to store the alignment results of the anchors. They will be used to align all other images.
    # All quantities are given from the reference to the anchor; first element corresponds to the reference and is thus left at 0.
    times = np.zeros(len(anchor_filenames)+1) # Elapsed time
    thetas = np.zeros(len(anchor_filenames)+1) # Rotation angle
    sun_moon_translations = np.zeros((len(anchor_filenames)+1, 2)) # Relative translation of the sun with respect to the moon 

    # Load ref image
    ref_img, ref_header = fits.read_fits_as_float(os.path.join(input_dir, ref_filename), checkstate=checkstate)
    # Moon preprocessing and detection
    ref_processed_img = registration.moon.preprocess(ref_img, num_clipped_pixels, img_callback=img_callback, checkstate=checkstate)
    ref_moon_center, ref_moon_radius = registration.moon.detect_moon(ref_processed_img, num_edge_pixels, smoothing, img_callback=img_callback, checkstate=checkstate)
    
    # Sun preprocessing
    ref_processed_img, ref_mass_center = registration.sun.preprocess(ref_processed_img, ref_moon_center, ref_moon_radius, img_callback=img_callback, checkstate=checkstate)
    
    # Save image
    ref_header = fits.update_header(ref_header, registration.moon.keyword_dict(*ref_moon_center, ref_moon_radius))
    fits.save_as_fits(ref_img, ref_header, os.path.join(moon_registered_dir, ref_filename), checkstate=checkstate)
    fits.save_as_fits(ref_img, ref_header, os.path.join(sun_registered_dir, ref_filename), checkstate=checkstate)

    for i, filename in enumerate(anchor_filenames, start=1):
        # Load anchor image
        img, header = fits.read_fits_as_float(os.path.join(input_dir, filename), checkstate=checkstate)
        # Moon preprocessing and detection
        processed_img = registration.moon.preprocess(img, num_clipped_pixels, img_callback=img_callback, checkstate=checkstate)
        moon_center, moon_radius = registration.moon.detect_moon(processed_img, num_edge_pixels, smoothing, img_callback=img_callback, checkstate=checkstate)
        
        # Sun preprocessing
        processed_img, _ = registration.sun.preprocess(processed_img, moon_center, moon_radius, img_callback=img_callback, checkstate=checkstate)
        # Compute transform parameters
        theta, tx, ty = registration.sun.compute_transform(ref_processed_img, processed_img, ref_mass_center, blend_factor, img_callback=img_callback, checkstate=checkstate)
        
        # Compute moon and sun transforms
        moon_tform = transform.centered_rigid_transform(ref_moon_center, theta, moon_center-ref_moon_center) # ref to anchor
        sun_tform = transform.centered_rigid_transform(ref_mass_center, theta, (tx,ty)) # ref to anchor
        
        # Apply transforms and save registered images
        moon_registered_img = transform.warp(img, moon_tform.inverse.params) # inverse required for anchor to ref
        sun_registered_img = transform.warp(img, sun_tform.inverse.params) # inverse required for anchor to ref
        moon_registered_header = fits.update_header(header, registration.moon.keyword_dict(*ref_moon_center, moon_radius))
        sun_registered_header = fits.update_header(header, registration.moon.keyword_dict(*sun_tform.inverse(moon_center)[0], moon_radius))
        fits.save_as_fits(moon_registered_img, moon_registered_header, os.path.join(moon_registered_dir, filename), checkstate=checkstate)
        fits.save_as_fits(sun_registered_img, sun_registered_header, os.path.join(sun_registered_dir, filename), checkstate=checkstate)

        # Update trackers
        times[i] = (datetime.strptime(header["DATE-OBS"], "%Y-%m-%dT%H:%M:%S") - datetime.strptime(ref_header["DATE-OBS"], "%Y-%m-%dT%H:%M:%S")).total_seconds()
        thetas[i] = theta
        sun_moon_translations[i] = registration.sun.compute_sun_moon_translation(sun_tform, moon_tform)

    theta_interp = scipy.interpolate.interp1d(times, thetas, kind='linear', fill_value='extrapolate')
    sun_moon_translation_interp = interpolation.LinearFitInterp(times, sun_moon_translations)

    _, _, filenames = next(os.walk(input_dir)) # not going into subfolders
    filenames = [f for f in filenames if f != ref_filename and f not in anchor_filenames] # remove ref and anchor
    for filename in filenames:
        # Load image
        img, header = fits.read_fits_as_float(os.path.join(input_dir, filename), checkstate=checkstate)
        # Moon preprocessing and detection
        processed_img = registration.moon.preprocess(img, num_clipped_pixels, img_callback=img_callback, checkstate=checkstate)
        moon_center, moon_radius = registration.moon.detect_moon(processed_img, num_edge_pixels, smoothing, img_callback=img_callback, checkstate=checkstate)
        
        # Interpolate transform parameters
        time = (datetime.strptime(header["DATE-OBS"], "%Y-%m-%dT%H:%M:%S") - datetime.strptime(ref_header["DATE-OBS"], "%Y-%m-%dT%H:%M:%S")).total_seconds()
        theta = theta_interp(time).item() # interp1d's call method always returns an array, even if the input is not one
        sun_moon_translation = sun_moon_translation_interp(time)

        # Compute moon and sun transforms
        moon_tform = transform.centered_rigid_transform(ref_moon_center, theta, moon_center-ref_moon_center)
        sun_tform = transform.translation_transform(sun_moon_translation) + moon_tform

        # Apply transforms and save registered images
        moon_registered_img = transform.warp(img, moon_tform.inverse.params) # inverse required for anchor to ref
        sun_registered_img = transform.warp(img, sun_tform.inverse.params) # inverse required for anchor to ref
        moon_registered_header = fits.update_header(header, registration.moon.keyword_dict(*ref_moon_center, moon_radius))
        sun_registered_header = fits.update_header(header, registration.moon.keyword_dict(*sun_tform.inverse(moon_center)[0], moon_radius))
        fits.save_as_fits(moon_registered_img, moon_registered_header, os.path.join(moon_registered_dir, filename), checkstate=checkstate)
        fits.save_as_fits(sun_registered_img, sun_registered_header, os.path.join(sun_registered_dir, filename), checkstate=checkstate)


if __name__ == "__main__":
    import sys
    from core.lib.utils import ColorTerminalStream
    sys.stdout = ColorTerminalStream()

    from core.parameters import INPUT_DIR, MOON_REGISTERED_DIR, SUN_REGISTERED_DIR, IMAGE_SCALE

    #REF_FILENAME = "0.01667s_2024-04-09_02h42m25s.fits"

    
    ref_filename = "0.25000s_2024-04-09_02h42m31s.fits"
    anchor_filenames = ["0.25000s_2024-04-09_02h40m33s.fits"]

    main(INPUT_DIR, ref_filename, anchor_filenames, MOON_REGISTERED_DIR, SUN_REGISTERED_DIR, IMAGE_SCALE)