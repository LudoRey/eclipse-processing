import os
from matplotlib import pyplot as plt
import numpy as np
from astropy.coordinates import EarthLocation
from astropy.time import Time
import scipy.interpolate
import scipy.ndimage

from lib.registration import translate, moon_detection, get_moon_radius, phase_correlation, highpass_filter, prep_for_rotation, polar_transform, fft_logmag, subpixel_maximum
from lib.fits import read_fits_as_float, save_as_fits
from lib.display import auto_ht_params, ht_lut, compute_statistics, center_crop
from lib.polar import warp_cart_to_polar
from lib.disk import binary_disk
from lib.filters import partial_filter

def auto_stf(img):
    stats = compute_statistics(img)
    stf_params = auto_ht_params(stats)
    img = ht_lut(np.ascontiguousarray(img), *stf_params)
    return img


def inpaint_interpolate(img: np.ndarray, mask: np.ndarray):
    h, w = img.shape
    inpainted_img = img.copy()

    # Create grid coordinates
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    
    # Get known pixel coordinates and values
    known_points = np.column_stack((x[~mask], y[~mask]))
    known_values = img[~mask]

    # Interpolate at masked points
    inpaint_values = scipy.interpolate.griddata(
        known_points, known_values, (x[mask], y[mask]), method='nearest'
    )

    # Fill in the masked regions
    inpainted_img[mask] = inpaint_values

    return inpainted_img

def main(input_dir,
         latitude,
         longitude,
         time_offset,
         image_scale,
         ref_filename,
         other_filename):

    # Load images
    img1, header1 = read_fits_as_float(os.path.join(input_dir, ref_filename))
    img2, header2 = read_fits_as_float(os.path.join(input_dir, other_filename))

    # Retrieve apparent moon radius
    location = EarthLocation(lat=latitude, lon=longitude, height=0)
    time = Time(header1["DATE-OBS"], scale='utc') - time_offset
    moon_radius_degree = get_moon_radius(time, location)
    moon_radius_pixels = moon_radius_degree * 3600 / image_scale

    # TODO
    saturation_values = np.array([0.138, 0.141, 0.136]) # (R,G,B) values
    saturation_value = 0.11

    fig, axes = plt.subplots(2, 3)

    imgs = [img1, img2]
    for i in range(2):
        if i == 0:
            moon_x, moon_y = 3205.55, 2185.47
        if i == 1:
            moon_x, moon_y = 3417.89, 2382.63
        moon_mask = binary_disk(moon_x, moon_y, moon_radius_pixels*1.05, imgs[i].shape)

        imgs[i] = center_crop(imgs[i], int(moon_x), int(moon_y), 2048, 2048)
        moon_mask = center_crop(moon_mask, int(moon_x), int(moon_y), 2048, 2048)

        imgs[i] = prep_for_rotation(imgs[i], moon_mask, saturation_value)
        axes[i,0].imshow(imgs[i])

        imgs[i] = fft_logmag(imgs[i])
        axes[i,1].imshow(imgs[i])

        imgs[i] = highpass_filter(imgs[i], frequency=0.01)
        axes[i,2].imshow(imgs[i])

        imgs[i] = polar_transform(imgs[i])
        #axes[i,2].imshow(imgs[i])

    # Compute cross-power spectrum
    f1 = np.fft.fft2(imgs[0])
    f2 = np.fft.fft2(imgs[1])
    cross_power_spectrum = (f1 * np.conj(f2)) / np.abs(f1 * np.conj(f2))
    
    # Find maximum response
    correlation_img = np.real(np.fft.ifft2(cross_power_spectrum))
    values = correlation_img[:, 0]
    print(values.shape)
    r = 2
    print(values[np.arange(-r, r+1)])
    angle = subpixel_maximum(values)
    print(angle)

    plt.show()

if __name__ == "__main__":

    from parameters import IMAGE_SCALE
    from parameters import TIME_OFFSET, LATITUDE, LONGITUDE
    from parameters import INPUT_DIR

    ref_filename = "0.25000s_2024-04-09_02h40m33s.fits"
    other_filename = "0.25000s_2024-04-09_02h42m31s.fits"

    main(INPUT_DIR,
         LATITUDE,
         LONGITUDE,
         TIME_OFFSET,
         IMAGE_SCALE,
         ref_filename,
         other_filename)