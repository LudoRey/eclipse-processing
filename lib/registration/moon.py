import numpy as np
import skimage as sk

from core.lib import display, filters
from core.lib.utils import cprint


def preprocess(img: np.ndarray, num_clipped_pixels: float, smoothing: float):
    'Rescale and clip pixels to make moon border more defined'
    # Convert to grayscale
    if len(img.shape) == 3:
        img = img.mean(axis=2)
    # Compute and apply threshold
    print(f"Clipping {num_clipped_pixels:.0f} pixels...", end=" ")
    quantile_threshold = 1 - num_clipped_pixels / img.size
    threshold = np.quantile(img, quantile_threshold)
    print(f"Threshold : {threshold:.4f}.")
    img = np.clip(img, 0, threshold)
    img /= threshold
    # Optional smoothing (in preparation for Sobel filtering)
    if smoothing:
        print(f"Applying smoothing with sigma = {smoothing:.1f}...")
        img = filters.gaussian_filter(img, sigma=smoothing)
    return img


def detect(img: np.ndarray, num_edge_pixels: float, return_img: bool=True):
    # Compute edge image
    print(f"Extracting {num_edge_pixels:.0f} edge pixels...")
    img = filters.sobel_grad_mag(img)
    # Compute threshold
    quantile_threshold = 1 - num_edge_pixels / img.size
    threshold = np.quantile(img, quantile_threshold)
    edges_coords = np.column_stack(np.nonzero(img > threshold))

    # RANSAC
    print("RANSAC circle fitting...", end=" ")
    min_samples = 3 # Number of random samples used to estimate the model parameters at each iteration
    residual_threshold = 1 # Inliers are such that |sqrt(x**2 + y**2) - r| < threshold. Might depend on pixel scale, but shouldnt really be lower than 1...
    max_trials = 100 # Number of RANSAC trials 
    
    model, inliers = sk.measure.ransac(
        edges_coords,
        sk.measure.CircleModel,
        min_samples=min_samples,
        residual_threshold=residual_threshold,
        max_trials=max_trials
    )

    y_c, x_c, radius = model.params
    print(f"Found {inliers.sum()} inliers.")
    cprint("Circle parameters :", f"- Center : ({x_c:.2f}, {y_c:.2f})", f"- Radius : {radius:.2f}", color="green", sep="\n")

    if return_img:
        img = display.normalize(img)
        img = np.stack([img]*3, axis=2)
        for i in range(3):
            img[edges_coords[inliers][:,0], edges_coords[inliers][:,1], i] = 1 if i == 1 else 0
            img[edges_coords[~inliers][:,0], edges_coords[~inliers][:,1], i] = 1 if i == 0 else 0
        return x_c, y_c, radius, img
    else:
        return x_c, y_c, radius
