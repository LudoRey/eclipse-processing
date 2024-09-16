import os
import numpy as np
from astropy.coordinates import EarthLocation
from astropy.time import Time

from lib.registration import translate, moon_detection, convert_angular_offset_to_x_y, get_sun_moon_offset, get_moon_radius
from lib.fits import read_fits_as_float, save_as_fits

def main(input_dir,
         moon_dir,
         sun_dir,
         latitude,
         longitude,
         time_offset,
         rotation,
         image_scale,
         ref_filename):
    
    os.makedirs(moon_dir, exist_ok=True)
    os.makedirs(sun_dir, exist_ok=True)
    
    location = EarthLocation(lat=latitude, lon=longitude, height=0)

    # Load reference image
    img, header = read_fits_as_float(os.path.join(input_dir, ref_filename))
    ref_time = Time(header["DATE-OBS"], scale='utc') - time_offset
    # Retrieve apparent moon radius
    moon_radius_degree = get_moon_radius(ref_time, location)
    moon_radius_pixels = moon_radius_degree * 3600 / image_scale
    # Compute reference moon center
    ref_moon_x, ref_moon_y, _ = moon_detection(img, moon_radius_pixels)
    # Compute reference sun center
    ref_delta_x, ref_delta_y = convert_angular_offset_to_x_y(*get_sun_moon_offset(ref_time, location), rotation, image_scale)
    ref_sun_x, ref_sun_y = ref_moon_x + ref_delta_x, ref_moon_y + ref_delta_y

    dirpath, _, filenames = next(os.walk(input_dir)) # not going into subfolders
    for filename in filenames:
        if filename == "0.25000s_2024-04-09_02h42m31s.fits" or filename == "0.25000s_2024-04-09_02h40m33s.fits":
        #if filename.endswith('.fits') and filename.startswith('0.00025s'):

            img, header = read_fits_as_float(os.path.join(dirpath, filename))

            # MOON ALIGNMENT
            if filename == ref_filename:
                moon_x, moon_y = ref_moon_x, ref_moon_y
                registered_img = img
            else:
                moon_x, moon_y, _ = moon_detection(img, moon_radius_pixels)
                registered_img = translate(img, ref_moon_x - moon_x, ref_moon_y - moon_y)
            # Update FITS keywords and save image
            header.set('MOON-X', ref_moon_x, 'X-coordinate of the moon center.')
            header.set('MOON-Y', ref_moon_y, 'Y-coordinate of the moon center.')
            header.set('TRANS-X', ref_moon_x - moon_x, 'X-translation applied during registration.')
            header.set('TRANS-Y', ref_moon_y - moon_y, 'Y-translation applied during registration.')
            save_as_fits(registered_img, header, os.path.join(moon_dir, filename))

            # SUN ALIGNMENT
            if filename == ref_filename:
                delta_x, delta_y = ref_delta_x, ref_delta_y
                sun_x, sun_y = ref_sun_x, ref_sun_y
                registered_img = img
            else:
                time = Time(header["DATE-OBS"], scale='utc') - time_offset
                delta_x, delta_y = convert_angular_offset_to_x_y(*get_sun_moon_offset(time, location), rotation, image_scale)
                sun_x, sun_y = moon_x + delta_x, moon_y + delta_y
                registered_img = translate(img, ref_sun_x - sun_x, ref_sun_y - sun_y)
            # Update FITS keywords and save image
            header.set('MOON-X', ref_sun_x - delta_x, 'X-coordinate of the moon center.')
            header.set('MOON-Y', ref_sun_y - delta_y, 'Y-coordinate of the moon center.')
            header.set('SUN-X', ref_sun_x, 'X-coordinate of the sun center.')
            header.set('SUN-Y', ref_sun_y, 'Y-coordinate of the sun center.')
            header.set('TRANS-X', ref_sun_x - sun_x, 'X-translation applied during registration.')
            header.set('TRANS-Y', ref_sun_y - sun_y, 'Y-translation applied during registration.')
            save_as_fits(registered_img, header, os.path.join(sun_dir, filename))

if __name__ == "__main__":

    from parameters import IMAGE_SCALE, ROTATION
    from parameters import TIME_OFFSET, LATITUDE, LONGITUDE
    from parameters import INPUT_DIR, MOON_DIR, SUN_DIR

    REF_FILENAME = "0.01667s_2024-04-09_02h42m25s.fits"

    main(INPUT_DIR,
         MOON_DIR,
         SUN_DIR,
         LATITUDE,
         LONGITUDE,
         TIME_OFFSET,
         ROTATION,
         IMAGE_SCALE,
         REF_FILENAME)