import numpy as np

from skimage.feature import canny
from skimage import measure, transform
from scipy import ndimage

from .display import center_crop
from .disk import binary_disk
from .polar import warp_cart_to_polar

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

def moon_detection(img, moon_radius_pixels):
    # Preparing image for detection
    if len(img.shape) == 3:
        # Convert to grayscale
        img = img.mean(axis=2)
    # Rescale and clip pixels to make moon border more defined
    # Because of brightness variations, the marging should be large enough so that a complete annulus is clipped
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

### The following code does not take spherical geometry into account!

# def get_sun_delta(time: Time, location: EarthLocation):
#     # Get the moon and sun coordinates at the specified time and location
#     moon_coords = get_body("moon", time, location)
#     sun_coords = get_body("sun", time, location)
#     # Compute the difference
#     delta_ra = (sun_coords.ra - moon_coords.ra).to(u.arcsec).value
#     delta_dec = (sun_coords.dec - moon_coords.dec).to(u.arcsec).value
#     print("Sun displacement relative to the moon :")
#     print(f'    - RA : {delta_ra:.2f}"')
#     print(f'    - DEC : {delta_dec:.2f}"')
#     return delta_ra, delta_dec

# def convert_ra_dec_to_x_y(ra, dec, rotation, image_scale):
#     # The rotation angle describes the rotation to go from [X,-Y] to [-RA,DEC] (tip: to see this, always place yourself in target coordinates)
#     # X: left -> right, RA: right -> left
#     # Y: up -> bottom, DEC: bottom -> up
#     # Rotation = 0 when RA = -X and DEC = -Y
#     ra_dec = np.array([ra, dec])
#     theta = - rotation * np.pi / 180 # /!\ Angle needs to be flipped
#     rotation_flip_matrix = np.array([[-np.cos(theta), -np.sin(theta)], # Rotation + flip (hence negative diagonal)
#                                     [np.sin(theta),  -np.cos(theta)]])
#     x_y = rotation_flip_matrix @ ra_dec / image_scale
#     # print("In image coordinates :")
#     # print(f'    - x : {x_y[0]:.2f}')
#     # print(f'    - y : {x_y[1]:.2f}')
#     return x_y[0], x_y[1]

# def convert_x_y_to_ra_dec(x, y, rotation, image_scale):
#     # The rotation angle describes the rotation to go from [X,-Y] to [-RA,DEC] (tip: to see this, always place yourself in target coordinates)
#     # X: left -> right, RA: right -> left
#     # Y: up -> bottom, DEC: bottom -> up
#     # Rotation = 0 when RA = -X and DEC = -Y
#     x_y = np.array([x, y])
#     theta = rotation * np.pi / 180 
#     rotation_flip_matrix = np.array([[-np.cos(theta), -np.sin(theta)], # Rotation + flip (hence negative diagonal)
#                                     [np.sin(theta),  -np.cos(theta)]])
#     ra_dec = rotation_flip_matrix @ x_y * image_scale
#     return ra_dec[0], ra_dec[1]

def phase_correlation(img1, img2):
    # Compute cross-power spectrum
    f1 = np.fft.fft2(img1)
    f2 = np.fft.fft2(img2)
    cross_power_spectrum = (f1 * np.conj(f2)) / np.abs(f1 * np.conj(f2))
    
    # Find maximum response
    impulse_img = np.fft.ifft2(cross_power_spectrum)
    dx, dy = np.unravel_index(np.argmax(np.abs(impulse_img)), impulse_img.shape) # (y,x) coords
    return dx, dy

def prep_for_rotation(img, moon_mask, saturation_value=0.11):
    print("Preparing image...")
    # Convert to grayscale
    if len(img.shape) == 3:
        img = img.mean(axis=2)

    # Mask out the moon and saturated pixels (set to a constant saturation_value)
    saturation_mask = img >= saturation_value
    mask = saturation_mask | moon_mask
    img[mask] = saturation_value
    # High-pass filter
    #img = img - ndimage.gaussian_filter(img, sigma=4)
    # Window to attenuate border discontinuities
    window = np.outer(np.hanning(img.shape[0]), np.hanning(img.shape[1])).reshape(img.shape)
    img = img*window

    return img

def fft_logmag(img):
    print("Computing FFT...")
    fft = np.fft.fft2(img, axes=[0,1])
    fft_mag = np.abs(fft)
    fft_mag = np.fft.fftshift(fft_mag)
    return np.log1p(fft_mag)

def highpass_filter(spectrum, frequency):
    M, N = spectrum.shape 
    u = np.linspace(-0.5, 0.5, M)
    v = np.linspace(-0.5, 0.5, N)
    U, V = np.meshgrid(u, v)

    D2 = U**2 + V**2
    H = 1 - np.exp(-D2 / (2*frequency**2))
    return H*spectrum

def polar_transform(img):
    # Polar transform centered in the middle of the image
    y_c, x_c = img.shape[0] // 2, img.shape[1] // 2
    output_shape = [3600, 3600]
    polar_img, theta_factor, rho_factor = warp_cart_to_polar(img, x_c, y_c, output_shape, return_factors=True, log_scaling=False)
    # Inscribed circle only (warp_cart_to_polar returns circumscribed)
    border_rho = min(img.shape[0] - y_c, img.shape[1] - x_c) 
    polar_img = polar_img[:,:int(np.ceil(border_rho*rho_factor))]
    return polar_img

def subpixel_maximum(values):
    # Initial guess
    x0 = np.argmax(values)
    # Quadratic fit
    x = x0 - (values[x0+1] - values[x0-1]) / (2*(values[x0+1] - 2*values[x0] + values[x0-1]))
    return x