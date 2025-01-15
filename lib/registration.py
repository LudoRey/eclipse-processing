import numpy as np

from skimage.feature import canny
from skimage import measure, transform

from astropy.coordinates import EarthLocation, get_body
from astropy.time import Time

def rotate_translate(img, theta, dx, dy):
    print(f"Rotating image by :")
    print(f"    - theta = {theta:.4f}")
    print(f"Translating image by :")
    print(f"    - dx = {dx:.2f}")
    print(f"    - dy = {dy:.2f}")
    tform = transform.AffineTransform(rotation=theta, translation=(dx, dy))
    registered_img = transform.warp(img, tform.inverse)
    return registered_img

def moon_detection(img, moon_radius_pixels):
    # Preparing image for detection
    if len(img.shape) == 3:
        # Convert to grayscale
        img = img.mean(axis=2)
    # Rescale and clip pixels to make moon border more defined
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
    
    # Canny
    print(f"Canny edge detection...")
    # Find the (appproximate) number of pixels that correspond to the moon circonference (M)
    # Canny will use a threshold that retains the M brightest pixels in the gradient image (before NMS, so there might be less of them after NMS)
    moon_circonference_pixels = 2*np.pi*moon_radius_pixels 
    moon_circonference_fraction = moon_circonference_pixels / img.size

    threshold = 1-moon_circonference_fraction # Single threshold : no hysteresis

    edges = canny(img, sigma=1, low_threshold=threshold, high_threshold=threshold, use_quantiles=True)
    print(f"Found {np.count_nonzero(edges)} edge pixels.")

    # RANSAC
    print("RANSAC fitting...")
    min_samples = 20 # Number of random samples used to estimate the model parameters at each iteration
    residual_threshold = 1 # Inliers are such that |sqrt(x**2 + y**2) - r| < threshold. Might depend on pixel scale, but shouldnt really be lower than 1...
    max_trials = 100 # Number of RANSAC trials 
    edges_coords = np.column_stack(np.nonzero(edges))
    model, inliers = measure.ransac(edges_coords, measure.CircleModel,
                                    min_samples=min_samples, residual_threshold=residual_threshold, max_trials=max_trials)

    y_c, x_c, radius = model.params
    print(f"Found {inliers.sum()} inliers.")
    print(f"Model parameters : ")
    print(f"    - x_c : {x_c:.2f}")
    print(f"    - y_c : {y_c:.2f}")
    print(f"    - radius : {radius:.2f}")

    return x_c, y_c, radius

def get_moon_radius(time: Time, location: EarthLocation):
    moon_coords = get_body("moon", time, location)
    earth_coords = get_body("earth", time, location)
    moon_dist_km = earth_coords.separation_3d(moon_coords).km
    moon_real_radius_km = 1737.4
    moon_radius_degree = np.arctan(moon_real_radius_km / moon_dist_km) * 180 / np.pi
    return moon_radius_degree

def get_sun_moon_offset(time: Time, location: EarthLocation):
    # Get the moon and sun coordinates at the specified time and location
    moon_coords = get_body("moon", time, location)
    sun_coords = get_body("sun", time, location)
    # Compute the offset
    sun_offset_scalar = moon_coords.separation(sun_coords).arcsecond
    sun_offset_angle = moon_coords.position_angle(sun_coords).degree
    return sun_offset_scalar, sun_offset_angle

def convert_angular_offset_to_x_y(offset_scalar, offset_angle, camera_rotation, image_scale):
    '''
    offset_scalar is given in arcseconds
    offset_angle and camera_rotation are given in degrees
    image_scale is given in arcseconds/pixel
    '''
    # 1) offset is offset_angle degrees east of north
    # 2) up (-y) is camera_rotation degrees east of north
    # Combining 1) and 2), offset is offset_angle + (360 - camera_rotation) degrees counterclockwise of up
    # Modulo 360, offset is offset_angle - camera_rotation + 90 degrees counterclockwise of x
    offset_angle_to_x = (offset_angle - camera_rotation + 90) * np.pi / 180 # counterclockwise
    offset_x = np.cos(offset_angle_to_x)*offset_scalar / image_scale
    offset_y = -np.sin(offset_angle_to_x)*offset_scalar / image_scale
    return offset_x, offset_y