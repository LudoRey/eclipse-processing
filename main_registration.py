import os
from matplotlib import pyplot as plt
from astropy.coordinates import EarthLocation
from astropy.time import Time

from registration import translate, moon_detection, prepare_img_for_detection, get_sun_delta, convert_ra_dec_to_x_y
from utils import read_fits_as_float, save_as_fits
from parameters import MOON_RADIUS_DEGREE
from parameters import SATURATION_VALUE, CLIP_EXP_TIME
from parameters import IMAGE_SCALE, ROTATION
from parameters import TIME_OFFSET, LATITUDE, LONGITUDE
from parameters import INPUT_DIR, MOON_DIR, SUN_DIR, REF_FILENAME, FILENAME


location = EarthLocation(lat=LATITUDE, lon=LONGITUDE, height=0)
moon_radius_pixels = MOON_RADIUS_DEGREE * 3600 / IMAGE_SCALE

os.makedirs(MOON_DIR, exist_ok=True)
os.makedirs(SUN_DIR, exist_ok=True)

# Compute reference moon center
img, header = read_fits_as_float(os.path.join(INPUT_DIR, REF_FILENAME))
ref_x_c, ref_y_c, _ = moon_detection(prepare_img_for_detection(img, header, CLIP_EXP_TIME, SATURATION_VALUE), moon_radius_pixels)
header.set('MOON-X', ref_x_c, 'X-coordinate of the moon center.')
header.set('MOON-Y', ref_y_c, 'Y-coordinate of the moon center.')
save_as_fits(img, header, os.path.join(MOON_DIR, REF_FILENAME))
# Compute reference sun-moon delta
ref_time = Time(header["DATE-OBS"], scale='utc') - TIME_OFFSET 
ref_delta_ra, ref_delta_dec = get_sun_delta(ref_time, location)
ref_delta_x, ref_delta_y = convert_ra_dec_to_x_y(ref_delta_ra, ref_delta_dec, ROTATION, IMAGE_SCALE)
header.set('SUN-X', ref_x_c + ref_delta_x, 'X-coordinate of the sun center.')
header.set('SUN-Y', ref_y_c + ref_delta_y, 'Y-coordinate of the sun center.')
save_as_fits(img, header, os.path.join(SUN_DIR, REF_FILENAME))

dirpath, _, filenames = next(os.walk(INPUT_DIR)) # not going into subfolders
for filename in filenames:
    #if filename == FILENAME:
    if filename.endswith('.fits') and filename != REF_FILENAME:
        # MOON ALIGNMENT
        img, header = read_fits_as_float(os.path.join(dirpath, filename))
        x_c, y_c, _ = moon_detection(prepare_img_for_detection(img, header, CLIP_EXP_TIME, SATURATION_VALUE), moon_radius_pixels)
        dx, dy = ref_x_c - x_c, ref_y_c - y_c
        registered_img = translate(img, dx, dy)
        header.set('MOON-X', ref_x_c, 'X-coordinate of the moon center.')
        header.set('MOON-Y', ref_y_c, 'Y-coordinate of the moon center.')
        save_as_fits(registered_img, header, os.path.join(MOON_DIR, filename))
        # SUN ALIGNMENT
        time = Time(header["DATE-OBS"], scale='utc') - TIME_OFFSET 
        delta_ra, delta_dec = get_sun_delta(time, location)
        delta_x, delta_y = convert_ra_dec_to_x_y(delta_ra, delta_dec, ROTATION, IMAGE_SCALE)
        dx, dy = dx + ref_delta_x - delta_x, dy + ref_delta_y - delta_y
        registered_img = translate(img, dx, dy)
        header.set('MOON-X', ref_x_c + ref_delta_x - delta_x, 'X-coordinate of the moon center.')
        header.set('MOON-Y', ref_y_c + ref_delta_y - delta_y, 'Y-coordinate of the moon center.')
        header.set('SUN-X', ref_x_c + ref_delta_x, 'X-coordinate of the sun center.')
        header.set('SUN-Y', ref_y_c + ref_delta_y, 'Y-coordinate of the sun center.')
        save_as_fits(registered_img, header, os.path.join(SUN_DIR, filename))