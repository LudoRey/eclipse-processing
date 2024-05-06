import numpy as np
import matplotlib.pyplot as plt

IMAGE_FILEPATH = "data\\totality\\merged_hdr\\hdr.fits"
MOON_MASK_FILEPATH = "data\\totality\\merged_hdr\\moon_mask.fits"

from utils import read_fits_as_float
from polar import angle_map, radius_map

def achf_kernel_at_ij(i, j, theta, rho, sigma, return_components=False):
    # arguments go from -2*sigma to 2*sigma
    rho_center = rho[i,j]
    theta_center = theta[i,j]
    delta_rho = rho_center - rho
    delta_theta = theta_center - theta
    delta_theta[delta_theta > np.pi] = 2*np.pi - delta_theta[delta_theta > np.pi] # to handle the periodicity (not the same as modulo)
    if return_components:
        radial = np.exp(-delta_rho**2/(2*sigma**2))
        tangential = np.exp(-(rho_center*delta_theta)**2/(2*sigma**2))
        return radial, tangential
    else:
        kernel = np.exp(-(delta_rho**2 + (rho_center*delta_theta)**2)/(2*sigma**2)) 
        return kernel
    
def crop_inset(img, crop_center, crop_radii, scale=4, border_value=np.nan, border_thickness=2):
    # Crop
    i_left, i_right = crop_center[0]-crop_radii[0], crop_center[0]+crop_radii[0]
    j_top, j_bottom = crop_center[1]-crop_radii[1], crop_center[1]+crop_radii[1]
    crop = img[i_left:i_right+1, j_top:j_bottom+1]
    # Crop border
    img[i_left-border_thickness:i_right+1+border_thickness, j_top-border_thickness:j_top] = border_value
    img[i_left-border_thickness:i_right+1+border_thickness, j_bottom+1:j_bottom+1+border_thickness] = border_value
    img[i_left-border_thickness:i_left, j_top-border_thickness:j_bottom+1+border_thickness] = border_value
    img[i_right+1:i_right+1+border_thickness, j_top-border_thickness:j_bottom+1+border_thickness] = border_value
    # Add inset
    inset = crop.repeat(scale,axis=0).repeat(scale,axis=1)
    img[-inset.shape[0]:, -inset.shape[1]:] = inset
    # Inset border 
    img[-inset.shape[0]:, -inset.shape[1]-border_thickness:-inset.shape[1]] = border_value
    img[-inset.shape[0]-border_thickness:-inset.shape[0], -inset.shape[1]:] = border_value

SIGMA = 10

# img, header = read_fits_as_float(IMAGE_FILEPATH)
# mask, _ = read_fits_as_float(MOON_MASK_FILEPATH)
# x_c, y_c = header["SUN-X"], header["SUN-Y"]
# img_gray = img.mean(axis=2)

shape = [1000, 1000]
x_c, y_c = 500, 500

theta, rho = angle_map(x_c, y_c, shape), radius_map(x_c, y_c, shape)

# kernel = np.zeros(shape)
# for i, j in zip(np.arange(100, 500, step=50), np.arange(100, 500, step=50)):
#     kernel += achf_kernel_at_ij(i, j, theta, rho, SIGMA)

fig, axes = plt.subplots(1,2)

axes[0].imshow(theta); axes[0].set_title('Angle')
axes[1].imshow(rho); axes[1].set_title('Radius')
# axes[2].imshow(kernel); axes[2].set_title('Kernel examples')
# axes[2].scatter(int(x_c), int(y_c), marker='x', color='white')

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