import numpy as np
import matplotlib.pyplot as plt

from utils import crop_inset
from polar import angle_map, radius_map

def achf_kernel_at_ij(i, j, theta, rho, sigma, return_components=False):
    # TODO: only compute values for -2*sigma to 2*sigma 
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