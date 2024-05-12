import os
import numpy as np
from astropy.coordinates import EarthLocation
from astropy.time import Time

from registration import translate, moon_detection, prepare_img_for_detection, get_sun_delta, convert_ra_dec_to_x_y
from utils import read_fits_as_float, save_as_fits
from parameters import MOON_RADIUS_DEGREE
from parameters import IMAGE_SCALE, ROTATION
from parameters import TIME_OFFSET, LATITUDE, LONGITUDE
from parameters import INPUT_DIR, MOON_DIR, SUN_DIR

REF_FILENAME = "0.01667s_2024-04-09_02h42m25s.fits"
CLIP_SOLAR_RADII = 1.3 # Because of brightness variations, the marging should be large enough so that a complete annulus is clipped

location = EarthLocation(lat=LATITUDE, lon=LONGITUDE, height=0)
moon_radius_pixels = MOON_RADIUS_DEGREE * 3600 / IMAGE_SCALE

min_clipped_pixels = np.pi*(CLIP_SOLAR_RADII**2-1)*moon_radius_pixels**2 # Annulus area

os.makedirs(MOON_DIR, exist_ok=True)
os.makedirs(SUN_DIR, exist_ok=True)

# Load reference image and check min_clipped_pixels
img, header = read_fits_as_float(os.path.join(INPUT_DIR, REF_FILENAME))
if min_clipped_pixels > img.size - np.pi*(1.2*moon_radius_pixels)**2:
    raise ValueError(f"The value of CLIP_SOLAR_RADII ({CLIP_SOLAR_RADII}) is too large for your FOV.")
# Compute reference moon center
ref_x_c, ref_y_c, _ = moon_detection(prepare_img_for_detection(img, min_clipped_pixels), moon_radius_pixels)
# Update FITS keywords and save image
header.set('MOON-X', ref_x_c, 'X-coordinate of the moon center.')
header.set('MOON-Y', ref_y_c, 'Y-coordinate of the moon center.')
header.set('TRANS-X', 0.0, 'X-translation applied during registration.')
header.set('TRANS-Y', 0.0, 'Y-translation applied during registration.')
save_as_fits(img, header, os.path.join(MOON_DIR, REF_FILENAME))

# Compute reference sun-moon delta
ref_time = Time(header["DATE-OBS"], scale='utc') - TIME_OFFSET 
ref_delta_ra, ref_delta_dec = get_sun_delta(ref_time, location)
ref_delta_x, ref_delta_y = convert_ra_dec_to_x_y(ref_delta_ra, ref_delta_dec, ROTATION, IMAGE_SCALE)
# Update FITS keywords and save image
header.set('SUN-X', ref_x_c + ref_delta_x, 'X-coordinate of the sun center.')
header.set('SUN-Y', ref_y_c + ref_delta_y, 'Y-coordinate of the sun center.')
save_as_fits(img, header, os.path.join(SUN_DIR, REF_FILENAME))

dirpath, _, filenames = next(os.walk(INPUT_DIR)) # not going into subfolders
for filename in filenames:
    if filename.endswith('.fits') and filename != REF_FILENAME:

        # MOON ALIGNMENT
        img, header = read_fits_as_float(os.path.join(dirpath, filename))
        x_c, y_c, _ = moon_detection(prepare_img_for_detection(img, min_clipped_pixels), moon_radius_pixels)
        dx, dy = ref_x_c - x_c, ref_y_c - y_c
        registered_img = translate(img, dx, dy)
        # Update FITS keywords and save image
        header.set('MOON-X', ref_x_c, 'X-coordinate of the moon center.')
        header.set('MOON-Y', ref_y_c, 'Y-coordinate of the moon center.')
        header.set('TRANS-X', dx, 'X-translation applied during registration.')
        header.set('TRANS-Y', dy, 'Y-translation applied during registration.')
        save_as_fits(registered_img, header, os.path.join(MOON_DIR, filename))

        # SUN ALIGNMENT
        time = Time(header["DATE-OBS"], scale='utc') - TIME_OFFSET 
        delta_ra, delta_dec = get_sun_delta(time, location)
        delta_x, delta_y = convert_ra_dec_to_x_y(delta_ra, delta_dec, ROTATION, IMAGE_SCALE)
        dx, dy = dx + ref_delta_x - delta_x, dy + ref_delta_y - delta_y
        registered_img = translate(img, dx, dy)
        # Update FITS keywords and save image
        header.set('MOON-X', ref_x_c + ref_delta_x - delta_x, 'X-coordinate of the moon center.')
        header.set('MOON-Y', ref_y_c + ref_delta_y - delta_y, 'Y-coordinate of the moon center.')
        header.set('SUN-X', ref_x_c + ref_delta_x, 'X-coordinate of the sun center.')
        header.set('SUN-Y', ref_y_c + ref_delta_y, 'Y-coordinate of the sun center.')
        header.set('TRANS-X', dx, 'X-translation applied during registration.')
        header.set('TRANS-Y', dy, 'Y-translation applied during registration.')
        save_as_fits(registered_img, header, os.path.join(SUN_DIR, filename))