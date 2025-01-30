import numpy as np
import skimage as sk

def preprocess(img, moon_radius_pixels):
    'Rescale and clip pixels to make moon border more defined'
    # Preparing image for detection
    if len(img.shape) == 3:
        # Convert to grayscale
        img = img.mean(axis=2)
    # Because of brightness variations, the margin should be large enough so that a complete annulus is clipped
    clip_annulus_outer_moon_radii = 1.3 # outer radius (in moon radii) of the annulus
    clip_annulus_area_pixels = np.minimum(img.size, np.pi*(clip_annulus_outer_moon_radii*moon_radius_pixels)**2) - np.pi*moon_radius_pixels**2
    # Compute clipping value
    hist, bin_edges = np.histogram(img, bins=10000)
    cumhist = np.cumsum(hist)
    clip_idx = np.nonzero(cumhist > img.size - clip_annulus_area_pixels)[0][0]
    clip_value = bin_edges[clip_idx]
    num_clipped_pixels = img.size - cumhist[clip_idx-1]
    print(f"Rescaling and clipping pixels above {clip_value:.3f} (clipped {num_clipped_pixels} pixels)")
    img = np.clip(img, 0, clip_value)
    img /= clip_value
    return img

def detect(img, moon_radius_pixels):
    # Canny
    print(f"Canny edge detection...")
    # Find the (appproximate) number of pixels that correspond to the moon circonference (M)
    # Canny will use a threshold that retains the M brightest pixels in the gradient image (before NMS, so there might be less of them after NMS)
    moon_circonference_pixels = 2*np.pi*moon_radius_pixels 
    moon_circonference_fraction = moon_circonference_pixels / img.size

    threshold = 1-moon_circonference_fraction # Single threshold : no hysteresis

    edges = sk.feature.canny(img, sigma=1, low_threshold=threshold, high_threshold=threshold, use_quantiles=True)
    print(f"Found {np.count_nonzero(edges)} edge pixels.")

    # RANSAC
    print("RANSAC fitting...")
    min_samples = 20 # Number of random samples used to estimate the model parameters at each iteration
    residual_threshold = 1 # Inliers are such that |sqrt(x**2 + y**2) - r| < threshold. Might depend on pixel scale, but shouldnt really be lower than 1...
    max_trials = 100 # Number of RANSAC trials 
    edges_coords = np.column_stack(np.nonzero(edges))
    model, inliers = sk.measure.ransac(edges_coords, sk.measure.CircleModel,
                                    min_samples=min_samples, residual_threshold=residual_threshold, max_trials=max_trials)

    y_c, x_c, radius = model.params
    print(f"Found {inliers.sum()} inliers.")
    print(f"Model parameters : ")
    print(f"    - x_c : {x_c:.2f}")
    print(f"    - y_c : {y_c:.2f}")
    print(f"    - radius : {radius:.2f}")

    return x_c, y_c, radius
