if __name__ == "__main__":
    from _import import add_project_root_to_path
    add_project_root_to_path()

import os
import numpy as np
import skimage as sk

from core.lib import registration, fits, display, filters
from core.lib.utils import cprint

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

        cprint("Preprocessing...", style='bold')
        # Convert to grayscale
        if len(img.shape) == 3:
            img = img.mean(axis=2)
        # Rescale and clip pixels to make moon border more defined
        print(f"Clipping {num_clipped_pixels:.0f} pixels...", end=" ", flush=True)
        img, threshold = clip_brightest_pixels(img, num_clipped_pixels)
        checkstate()
        img_callback(img)
        print(f"Threshold : {threshold:.4f}.")
        # Compute image gradient (edges)
        print(f"Computing edge map...", end=" ", flush=True)
        img = compute_edge_map(img, smoothing)
        checkstate()
        img_callback(img)
        print("Done.")

        cprint("RANSAC circle fitting...", style='bold')
        edge_coords = extract_brightest_pixels(img, num_edge_pixels)
        (moon_x, moon_y, moon_radius), inliers = ransac_circle_fit(edge_coords)
        checkstate()
        img_callback(make_ransac_img(img, edge_coords, inliers))
        print(f"Found {inliers.sum()} inliers.")
        print("Circle parameters :")
        cprint(f"- Center : ({moon_x:.2f}, {moon_y:.2f})", f"- Radius : {moon_radius:.2f}", color="green", sep="\n")
        
        fits.update_fits_header(os.path.join(dirpath, filename),
                                {'MOON-X': (moon_x, 'X-coordinate of the moon center.'),
                                'MOON-Y': (moon_y, 'Y-coordinate of the moon center.'),
                                'MOON-R': (moon_radius, 'Radius of the moon.')})
        checkstate()

def compute_edge_map(img: np.ndarray, smoothing: float=None):
    # Optional smoothing (in preparation for Sobel filtering)
    if smoothing:
        img = filters.gaussian_filter(img, sigma=smoothing)
    # Compute Sobel gradient
    img = filters.sobel_grad_mag(img)
    return display.normalize(img)

def clip_brightest_pixels(img: np.ndarray, num_clipped_pixels: float):
    quantile_threshold = 1 - num_clipped_pixels / img.size
    threshold = np.quantile(img, quantile_threshold)
    img = np.clip(img, 0, threshold) / threshold
    return img, threshold

def extract_brightest_pixels(img: np.ndarray, num_edge_pixels: float):
    # Compute threshold
    quantile_threshold = 1 - num_edge_pixels / img.size
    threshold = np.quantile(img, quantile_threshold)
    # Extract and return coords
    return np.column_stack(np.nonzero(img > threshold)) # (N,2)

def ransac_circle_fit(pixel_coords: np.ndarray):
    # RANSAC parameters
    min_samples = 3 # Number of random samples used to estimate the model parameters at each iteration
    residual_threshold = 1 # Inliers are such that |sqrt(x**2 + y**2) - r| < threshold. Might depend on pixel scale, but shouldnt really be lower than 1...
    max_trials = 100 # Number of RANSAC trials 
    
    model, inliers = sk.measure.ransac(
        pixel_coords,
        sk.measure.CircleModel,
        min_samples=min_samples,
        residual_threshold=residual_threshold,
        max_trials=max_trials
    )
    return model.params, inliers

def make_ransac_img(img: np.ndarray, pixel_coords: np.ndarray, inliers: np.ndarray):
    img = np.stack([img]*3, axis=2)
    for i in range(3):
        img[pixel_coords[inliers][:,0], pixel_coords[inliers][:,1], i] = 1 if i == 1 else 0
        img[pixel_coords[~inliers][:,0], pixel_coords[~inliers][:,1], i] = 1 if i == 0 else 0
    return img

if __name__ == "__main__":
    import sys
    from core.lib.utils import ColorTerminalStream
    sys.stdout = ColorTerminalStream()

    from core.parameters import IMAGE_SCALE
    from core.parameters import MEASURED_TIME, UTC_TIME, LATITUDE, LONGITUDE
    from core.parameters import INPUT_DIR

    REF_FILENAME = "0.01667s_2024-04-09_02h42m25s.fits"

    main(INPUT_DIR, IMAGE_SCALE)