import os
from matplotlib import pyplot as plt
import numpy as np
import scipy.optimize

from lib.optimization import mse_rigid_registration_func_and_grad
from lib.registration import rotate_translate, moon_detection, get_moon_radius
from lib.phasecorr import correlation, prep_for_registration, get_moon_clipping_value
from lib.fits import read_fits_as_float, save_as_fits
from lib.display import auto_ht_params, ht_lut, compute_statistics, center_crop, crop


def main(input_dir, ref_filename, other_filenames):
    
    # Load reference image
    ref_img, ref_header = read_fits_as_float(os.path.join(input_dir, ref_filename))
    # TODO Center on the moon and crop 
    moon_x, moon_y = ref_header["MOON-X"], ref_header["MOON-Y"]
    w, h = 2048, 2048
    ref_img, ref_header = center_crop(ref_img, int(moon_x), int(moon_y), w, h, ref_header)
    # Get clipping value (for all images)
    clipping_value = get_moon_clipping_value(ref_img, ref_header) # Possible bug : dead pixels
    # Prepare image for registration
    ref_img = prep_for_registration(ref_img, ref_header, clipping_value)

    for filename in other_filenames:
        # Load image
        img, header = read_fits_as_float(os.path.join(input_dir, filename))
        # TODO Center on the moon and crop 
        moon_x, moon_y = header["MOON-X"], header["MOON-Y"]
        w, h = 2048, 2048
        img, header = center_crop(img, int(moon_x), int(moon_y), w, h, header)
        # Prepare image for registration
        img = prep_for_registration(img, header, clipping_value)

        # Compute cross-correlation between img and ref_img
        # The highest peak minimizes the MSE w.r.t. translation ref_img -> img
        correlation_img = correlation(img, ref_img)
        dy, dx = np.unravel_index(np.argmax(correlation_img), correlation_img.shape)
        dy = dy if dy <= h // 2 else dy - h # dy in [0,h-1] -> [-h//2+1, h//2]
        dx = dx if dx <= w // 2 else dx - w
        # We use it as an initial guess for the optimization-based approach, with theta = 0
        x0 = np.array([0, dx, dy])

        # Optimization-based registration
        fun = lambda x: mse_rigid_registration_func_and_grad(ref_img, img, x[0], x[1], x[2])

        def callback(x):
            print(x)
            f, g = fun(x)
            print(f, g)

        callback(x0)
        result = scipy.optimize.minimize(fun, x0, jac=True, options={'disp': True})

        theta, dx_sub, dy_sub = result.x # estimated parameters of the transform ref_img -> img
        print(result.message, "\n", result.status)

    fig, axes = plt.subplots(2, 2)

    axes[0,0].imshow(ref_img, vmin=-0.05, vmax=0.05)
    axes[1,0].imshow(img, vmin=-0.05, vmax=0.05)

    # Centering
    h, w = correlation_img.shape
    extent = [-w//2, w//2-1, h//2 - 1, -h//2]
    axes[0,1].imshow(np.fft.fftshift(correlation_img), extent=extent)
    
    # Zoom
    r = 4
    rolled_img = np.roll(np.roll(correlation_img, -dy+r, axis=0), -dx+r, axis=1)[:2*r+1, :2*r+1]
    axes[1,1].imshow(rolled_img, extent=[dx-r, dx+r, dy+r, dy-r])

    plt.show()

if __name__ == "__main__":

    from parameters import IMAGE_SCALE
    from parameters import MEASURED_TIME, UTC_TIME, LATITUDE, LONGITUDE
    from parameters import INPUT_DIR

    ref_filename = "0.25000s_2024-04-09_02h40m33s.fits"
    other_filenames = ["0.25000s_2024-04-09_02h42m31s.fits"]

    # ref_filename = "0.00100s_2024-04-09_02h39m59s.fits"
    # other_filenames = ["0.00100s_2024-04-09_02h43m02s.fits"]

    main(INPUT_DIR,
         ref_filename,
         other_filenames)