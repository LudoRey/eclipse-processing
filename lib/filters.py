import numpy as np
from .utils import Timer
from scipy.ndimage import gaussian_filter

def achf_kernel_at_ij(i, j, theta, rho, sigma, sun_radius=141.4213562373095, return_components=False):
    rho_center = rho[i,j]
    theta_center = theta[i,j]
    delta_rho = rho_center - rho
    delta_theta = theta_center - theta
    delta_theta[delta_theta > np.pi] = 2*np.pi - delta_theta[delta_theta > np.pi] # to handle the periodicity (not the same as modulo)
    ###
    scaling = rho_center # vanilla ACHF
    # scaling = sun_radius # uniform tangential component
    scaled_delta_theta = scaling*delta_theta
    ###

    radial = np.exp(-delta_rho**2/(2*sigma**2))
    tangential = np.exp(-(scaled_delta_theta)**2/(2*sigma**2))
    if return_components:
        return radial, tangential
    else:
        return radial*tangential
    

def new_achf_kernel_at_ij(i, j, theta, rho, sigma, sun_radius=141.4213562373095):
    # only compute values for -2*sigma to 2*sigma
    # still very slow

    rho_center = rho[i,j]
    theta_center = theta[i,j]
    delta_rho = rho_center - rho
    delta_theta = theta_center - theta
    delta_theta[delta_theta > np.pi] = 2*np.pi - delta_theta[delta_theta > np.pi] # to handle the periodicity (not the same as modulo)
    ###
    #scaling = rho_center # vanilla ACHF
    scaling = sun_radius # uniform tangential component
    scaled_delta_theta = scaling*delta_theta
    ###

    mask = (np.abs(delta_rho) < 2*sigma)*(np.abs(scaled_delta_theta) < 2*sigma)
    delta_rho_values = delta_rho[mask]
    scaled_delta_theta_values = scaled_delta_theta[mask]
    kernel_values = np.exp(-(delta_rho_values**2+scaled_delta_theta_values**2)/(2*sigma**2))
    kernel = np.zeros(mask.shape)
    kernel[mask == 1] = kernel_values
    return kernel

def achf(img_polar, sigma, j_0, rho_factor, theta_factor):
    blurred_img_polar = np.copy(img_polar)
    for j in range(j_0, blurred_img_polar.shape[1]): # we don't need to apply tangential blurring on j < j_0, because the mask is 0 there
        rho = j/rho_factor
        blurred_img_polar[:,j] = gaussian_filter(blurred_img_polar[:,j], sigma=(sigma/rho)*theta_factor, mode='wrap') # tangential
    blurred_img_polar = gaussian_filter(blurred_img_polar, sigma=sigma*rho_factor, axes=(1,), mode='reflect') # radial
    return blurred_img_polar

def radial_tangential(img_polar, sigma, rho_0, rho_factor, theta_factor):
    blurred_img_polar = gaussian_filter(img_polar, sigma=(sigma/rho_0)*theta_factor, axes=(0,), mode='wrap') # tangential
    blurred_img_polar = gaussian_filter(blurred_img_polar, sigma=sigma*rho_factor, axes=(1,), mode='reflect') # radial
    return blurred_img_polar

def partial_filter(img, mask, filter_func, filter_args):
    print("Applying partial filter...")
    # A partial convolution of an image I by a kernel K and with weights W can be computed as the division of two normal convolutions.
    # Compute the numerator : conv(W*I, K)
    blurred_img = filter_func(mask * img, **filter_args) 
    # Compute the denominator : conv(W, K)
    blurred_mask = filter_func(mask, **filter_args)
    # Get the partial convolution (divide the numerator by the denominator)
    blurred_img[mask != 0] = blurred_img[mask != 0] / blurred_mask[mask != 0] # if mask == 0, blurred_img will be set the 0 (see below)
    # Set blurred image to 0 if W = 0
    blurred_img = mask*blurred_img
    return blurred_img

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from polar import angle_map, radius_map
    from utils import crop_inset

    SIGMA = 10

    shape = [1000, 1000]
    x_c, y_c = 500, 500

    theta, rho = angle_map(x_c, y_c, shape), radius_map(x_c, y_c, shape)

    fig, axes = plt.subplots(1,2)

    axes[0].imshow(theta); axes[0].set_title('Angle')
    axes[1].imshow(rho); axes[1].set_title('Radius')

    fig1, axes1 = plt.subplots(2,3)

    i = [200, 400]
    j = [200, 400]
    for k in range(2):
        radial, tangential = achf_kernel_at_ij(i[k], j[k], theta, rho, SIGMA, return_components=True)
        crop_inset(radial, [i[k],j[k]], [50,50])
        crop_inset(tangential, [i[k],j[k]], [50,50])
        axes1[k,0].imshow(radial); axes1[0,0].set_title("Radial component")
        axes1[k,1].imshow(tangential); axes1[0,1].set_title("Tangential component")
        axes1[k,2].imshow(radial*tangential); axes1[0,2].set_title("Kernel")

    plt.show()