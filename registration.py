import numpy as np

from skimage.feature import canny
from skimage import measure, transform

from astropy.coordinates import EarthLocation, get_body
from astropy.time import Time
from astropy import units as u

def translate(img, dx, dy):
    print(f"Translating image by :")
    print(f"    - dx = {dx:.2f}")
    print(f"    - dy = {dy:.2f}")
    tform = transform.AffineTransform(translation=(dx, dy))
    registered_img = transform.warp(img, tform.inverse)
    return registered_img

def prepare_img_for_detection(img, header, clip_exp_time, saturation_value):
    if len(img.shape) == 3:
        # Convert to grayscale
        img = img.mean(axis=2)
    if "PEDESTAL" in header: # TODO : need better handling of the pedestal / saturation relationship here
        # Remove pedestal
        img = img - header["PEDESTAL"] / 65535
    if header["EXPTIME"] < clip_exp_time:
        # Artificially clip short exposures to facilitate edge detection
        img *= clip_exp_time/header["EXPTIME"]
        img = np.clip(img, 0, saturation_value)
    return img

def moon_detection(img, moon_radius_pixels):
    # Canny
    print(f"Canny edge detection...")
    # Find the (appproximate) number of pixels that correspond to the moon circonference (moon_circonference_pixels, or M)
    # Canny will use a high threshold that retains the M brightest pixels in the gradient image 
    moon_circonference_pixels = 2*np.pi*moon_radius_pixels 
    moon_circonference_fraction = moon_circonference_pixels / img.size

    low_threshold = 1-moon_circonference_fraction
    high_threshold = 1-moon_circonference_fraction
    #print(f"    - Setting high threshold that corresponds to the {int(moon_circonference_pixels)} brightest edge pixels before NMS.")

    edges = canny(img, sigma=1, low_threshold=low_threshold, high_threshold=high_threshold, use_quantiles=True)
    print(f"Found {np.count_nonzero(edges)} edge pixels.")

    # RANSAC
    print("RANSAC fitting...")
    min_samples = 20 # Number of random samples used to estimate the model parameters at each iteration
    residual_threshold = 1 # Inliers are such that |sqrt(x**2 + y**2) - r| < threshold. Might depend on pixel scale ? But shouldnt really be lower than 1...
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

def get_sun_delta(time: Time, location: EarthLocation):
    # Get the moon and sun coordinates at the specified time and location
    moon_coords = get_body("moon", time, location)
    sun_coords = get_body("sun", time, location)
    # Compute the difference
    delta_ra = (sun_coords.ra - moon_coords.ra).to(u.arcsec).value
    delta_dec = (sun_coords.dec - moon_coords.dec).to(u.arcsec).value
    print("Sun displacement relative to the moon :")
    print(f'    - RA : {delta_ra:.2f}"')
    print(f'    - DEC : {delta_dec:.2f}"')
    return delta_ra, delta_dec

def convert_ra_dec_to_x_y(ra, dec, rotation, image_scale):
    # The rotation angle describes the rotation to go from [X,-Y] to [-RA,DEC] (tip: to see this, always place yourself in target coordinates)
    # X: left -> right, RA: right -> left
    # Y: up -> bottom, DEC: bottom -> up
    # Rotation = 0 when RA = -X and DEC = -Y
    ra_dec = np.array([ra, dec])
    theta = - rotation * np.pi / 180 # /!\ Angle needs to be flipped
    rotation_flip_matrix = np.array([[-np.cos(theta), -np.sin(theta)], # Rotation + flip (hence negative diagonal)
                                    [np.sin(theta),  -np.cos(theta)]])
    x_y = rotation_flip_matrix @ ra_dec / image_scale
    # print("In image coordinates :")
    # print(f'    - x : {x_y[0]:.2f}')
    # print(f'    - y : {x_y[1]:.2f}')
    return x_y[0], x_y[1]

def convert_x_y_to_ra_dec(x, y, rotation, image_scale):
    # The rotation angle describes the rotation to go from [X,-Y] to [-RA,DEC] (tip: to see this, always place yourself in target coordinates)
    # X: left -> right, RA: right -> left
    # Y: up -> bottom, DEC: bottom -> up
    # Rotation = 0 when RA = -X and DEC = -Y
    x_y = np.array([x, y])
    theta = rotation * np.pi / 180 
    rotation_flip_matrix = np.array([[-np.cos(theta), -np.sin(theta)], # Rotation + flip (hence negative diagonal)
                                    [np.sin(theta),  -np.cos(theta)]])
    ra_dec = rotation_flip_matrix @ x_y * image_scale
    return ra_dec[0], ra_dec[1]