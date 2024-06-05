import numpy as np
from skimage.transform import warp

def angle_map(x_c, y_c, shape):
    x, y = np.arange(shape[1]), np.arange(shape[0])
    X, Y = np.meshgrid(x, y)
    with np.errstate(divide='ignore',invalid='ignore'):
        theta = np.arctan((Y-y_c)/(X-x_c)) + np.pi * (X-x_c < 0) + 2*np.pi * (X-x_c >= 0)*(Y-y_c < 0)  # tan is pi-periodic : arctan can be many things
        theta[np.isnan(theta)] = 0 # if x_c and y_c are integers, we get arctan(0/0) = NaN at the center
    return theta

def radius_map(x_c, y_c, shape):
    x, y = np.arange(shape[1]), np.arange(shape[0])
    X, Y = np.meshgrid(x, y)
    rho = np.sqrt((X-x_c)**2 + (Y-y_c)**2)
    return rho

def coords_cart_to_polar(cart_coords, x_c, y_c, theta_factor, rho_factor):
    '''
    To be used as the inverse_map argument of the warp function.
    The input cart_coords is a (M, 2) array where each row contains (x,y) coordinates.
    Returns a (M, 2) array where each row contains scaled (rho, theta) coordinates.
    '''
    x, y = cart_coords[:, 0], cart_coords[:, 1]
    rho = np.sqrt((x-x_c)**2 + (y-y_c)**2)
    with np.errstate(divide='ignore',invalid='ignore'):
        theta = np.arctan((y-y_c)/(x-x_c)) + np.pi * (x-x_c < 0) + 2*np.pi * (x-x_c >= 0)*(y-y_c < 0) # tan is pi-periodic : arctan can be many things
        theta[np.isnan(theta)] = 0
    polar_coords = np.column_stack((rho*rho_factor, theta*theta_factor))
    return polar_coords

def coords_polar_to_cart(polar_coords, x_c, y_c, theta_factor, rho_factor):
    '''
    To be used as the inverse_map argument of the warp function.
    The input polar_coords is a (M, 2) array where each row contains scaled (rho, theta) coordinates.
    Returns a (M, 2) array where each row contains (x, y) coordinates.
    '''
    rho, theta = polar_coords[:, 0] / rho_factor, polar_coords[:, 1] / theta_factor
    x = rho * np.cos(theta) + x_c
    y = rho * np.sin(theta) + y_c # minus sign because y is reversed in image coordinates
    cart_coords = np.column_stack((x, y))
    return cart_coords

def warp_cart_to_polar(img, x_c, y_c, output_shape, order=1, return_factors=False):
    # Compute maximum/minimum distance to the center
    corner_pts = np.array([[0, 0],[0, img.shape[1]],[img.shape[0], 0], [img.shape[0], img.shape[1]]])
    corner_dist = np.sqrt(np.sum((corner_pts - np.array([y_c, x_c]))**2, axis=1)) # (4) array
    max_rho = corner_dist.max()
    # Define scaling factors
    theta_factor = output_shape[0] / (2 * np.pi)
    rho_factor = output_shape[1] / max_rho 
    # Warp image
    print("Warping image to polar coordinates...")
    warp_args = {'x_c': x_c, 'y_c': y_c, 'theta_factor': theta_factor, 'rho_factor': rho_factor}
    warped_img = warp(img, inverse_map=coords_polar_to_cart, map_args=warp_args, output_shape=output_shape, mode='reflect', order=order)
    if return_factors:
        return warped_img, theta_factor, rho_factor 
    else:
        return warped_img

def warp_polar_to_cart(img, x_c, y_c, output_shape, order=1):
    # Compute maximum distance to the center
    corner_pts = np.array([[0, 0],[0, output_shape[1]],[output_shape[0], 0], [output_shape[0], output_shape[1]]])
    corner_dist = np.sqrt(np.sum((corner_pts - np.array([y_c, x_c]))**2, axis=1))
    max_rho = corner_dist.max()
    # Define scaling factors
    theta_factor = img.shape[0] / (2 * np.pi)
    rho_factor = img.shape[1] / max_rho
    # Warp image
    print("Warping image to cartesian coordinates...")
    warp_args = {'x_c': x_c, 'y_c': y_c, 'theta_factor': theta_factor, 'rho_factor': rho_factor}
    warped_img = warp(img, inverse_map=coords_cart_to_polar, map_args=warp_args, output_shape=output_shape, mode='wrap', order=order) # wrap padding for 0 = 2pi
    return warped_img