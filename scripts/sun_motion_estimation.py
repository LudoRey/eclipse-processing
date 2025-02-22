if __name__ == "__main__":
    from _import import add_project_root_to_path
    add_project_root_to_path()

import os
import numpy as np
import cv2
import skimage as sk
from datetime import datetime

#from core.lib.registration import sun, optim, transform, utils
from core.lib import fits, display, registration, transform, optim, disk, filters
from core.lib.utils import cprint


def main(input_dir,
         ref_filename,
         other_filename,
         *,
         blend_factor=0.75,
         img_callback=lambda img: None,
         checkstate=lambda: None):

    # Load images
    ref_img, ref_header = fits.read_fits_as_float(os.path.join(input_dir, ref_filename))
    img, header = fits.read_fits_as_float(os.path.join(input_dir, other_filename))
    # Convert to grayscale float32
    ref_img = ref_img.astype(np.float32).mean(axis=2)
    img = img.astype(np.float32).mean(axis=2)
    checkstate()
    img_callback(display.red_cyan_blend(ref_img, img, blend_factor=blend_factor))

    fmt = "%Y-%m-%dT%H:%M:%S"
    # Convert strings to datetime objects
    dt = datetime.strptime(header["DATE-OBS"], fmt)
    ref_dt = datetime.strptime(ref_header["DATE-OBS"], fmt)
    # Compute the difference in seconds
    timedelta = (dt - ref_dt).total_seconds()
    print(timedelta)

    cprint("Preprocessing:", style='bold')
    # Get clipping value
    print("Computing clipping value...", end=" ", flush=True)
    clipping_value = min(get_clipping_value(img, header) for img, header in zip([ref_img, img], [ref_header, header]))
    checkstate()
    print(f"{clipping_value:.4f}")

    # Hide moon
    print("Hiding moon...", end=" ", flush=True)
    ref_img, ref_mass_center = hide_moon(ref_img, ref_header, clipping_value)
    img, mass_center = hide_moon(img, header, clipping_value)
    checkstate()
    img_callback(display.red_cyan_blend(ref_img, img, blend_factor=blend_factor))
    print("Done.")

    # Preprocess image to register
    print("Applying bandpass filter...", end=" ", flush=True)
    ref_img = apply_bandpass_filter(ref_img, ref_mass_center)
    img = apply_bandpass_filter(img, mass_center)
    checkstate()
    img_callback(display.red_cyan_blend(ref_img, img, blend_factor=blend_factor))
    print("Done.")
    
    cprint("Initializing transform:", style='bold')
    # Initialize transform parameters
    print("Computing correlation peak...", end=" ", flush=True)
    tx, ty = registration.correlation_peak(img, ref_img) # translation ref_img -> img
    checkstate()
    print(f"({tx}, {ty})")
    # tx, ty = mass_center[0] - ref_mass_center[0], mass_center[1] - ref_mass_center[1]
    theta = 0
    rotation_center = ref_mass_center
    print(f"Using center of mass as rotation center : ({rotation_center[0]:.2f}, {rotation_center[1]:.2f})")

    def optim_callback(iter, x, delta, f, g=None):
        checkstate()
        # GUI callback
        theta, tx, ty = obj.convert_x_to_params(x)
        inv_tform = transform.centered_rigid_transform(center=rotation_center, rotation=theta, translation=(tx,ty))
        img_callback(display.red_cyan_blend(ref_img, transform.warp(img, inv_tform.inverse.params), blend_factor=blend_factor))

        # Display info
        if iter == 0:
            cprint("Optimizing transform:", style='bold')
        print(f"Iteration {iter}:")
        print(f"- Angle          : {x[0]:>9.3f} deg" + (f" ({delta[0]:+.3f})" if delta is not None else "")) 
        print(f"- Translation (x): {x[1]:>9.2f} px " + (f" ({delta[1]:+.2f})" if delta is not None else ""))  
        print(f"- Translation (y): {x[2]:>9.2f} px " + (f" ({delta[2]:+.2f})" if delta is not None else ""))
        print(f"- Objective value: {f:.3e}")
        #print(f"Objective gradient: {g[0]:.3e}, {g[1]:.3e}, {g[2]:.3e}")

    obj = registration.RigidRegistrationObjective(ref_img, img, rotation_center, theta_factor=180/np.pi)
    delta_max = 1
    delta_min = np.array([1e-4, 1e-3, 1e-3]) # want more precision on the angle
    x = optim.line_search_newton(obj.convert_params_to_x(theta, tx, ty),
                                 obj.value, obj.grad, obj.hess,
                                 delta_max=delta_max, delta_min=delta_min,
                                 callback=optim_callback)
    theta, tx, ty = obj.convert_x_to_params(x)


def get_clipping_value(img, header):
    # Find clipping value that surrounds the 1.05R moon mask
    moon_mask = disk.binary_disk(header["MOON-X"], header["MOON-Y"], header["MOON-R"]*1.05, img.shape)
    moon_mask_border = sk.morphology.binary_dilation(moon_mask) & ~moon_mask
    clipping_value = np.min(img[moon_mask_border]) # Possible bug : dead pixels
    return clipping_value

def hide_moon(img, header, clipping_value):
    # Clip the moon and its surroundings
    moon_mask = disk.binary_disk(header["MOON-X"], header["MOON-Y"], header["MOON-R"]*1.05, img.shape)
    # moon_mask_border = sk.morphology.binary_dilation(moon_mask) & ~moon_mask
    # clipping_value = np.min(img[moon_mask_border]) # Possible bug : dead pixels
    clipping_mask = img >= clipping_value # should surround the moon_mask
    mask = clipping_mask | moon_mask
    img[mask] = clipping_value
    moments = cv2.moments(mask.astype(np.int32), binaryImage=True)
    mass_center = (moments["m10"] / moments["m00"], moments["m01"] / moments["m00"])
    return img, mass_center

def apply_bandpass_filter(img, center, sigma_high_pass_tangential=10, sigma_low_pass=3):
    # Tangential high-pass filter
    img = img - filters.tangential_filter(img, center, sigma=sigma_high_pass_tangential)
    # Low-pass filter to match the bilinear interpolation smoothing that happens during registration
    img = filters.gaussian_filter(img, sigma=sigma_low_pass)
    # Normalize
    img /= img.std()
    return img

# # Inpaint stars and hot pixels
# print("Creating DoG cube")
# dog_cube, sigma_list = get_dog_cube(img, 0.5, 2)

# print("Finding maxima")
# peaks = sk.feature.peak_local_max(dog_cube, threshold_abs=0.03, footprint=np.ones((3,)*dog_cube.ndim), exclude_border=False)

# print("Creating mask")
# mask = np.zeros_like(img, dtype=bool)
# for i, sigma in enumerate(sigma_list[:-1]):
#     temp_mask = np.zeros_like(img, dtype=bool)
#     peak_indices = (peaks[:,2] == i)
#     temp_mask[peaks[peak_indices,0], peaks[peak_indices,1]] = True
#     footprint = sk.morphology.disk(int(np.ceil(5*sigma)))
#     temp_mask = sk.morphology.binary_dilation(temp_mask, footprint)
#     mask = mask | temp_mask

# print("Inpainting")
# img = sk.restoration.inpaint_biharmonic(img, mask)

# def get_dog_cube(img, min_sigma=0.5, max_sigma=2, sigma_ratio=1.6):
#     # k such that min_sigma*(sigma_ratio**k) > max_sigma
#     k = int(np.mean(np.log(max_sigma / min_sigma) / np.log(sigma_ratio) + 1))

#     # a geometric progression of standard deviations for gaussian kernels
#     sigma_list = np.array([min_sigma * (sigma_ratio**i) for i in range(k + 1)])

#     # computing difference between two successive Gaussian blurred images
#     # to obtain an approximation of the scale invariant Laplacian of the
#     # Gaussian operator
#     dog_image_cube = np.empty(img.shape + (k,))
#     gaussian_previous = ndi.gaussian_filter(img, sigma=sigma_list[0])
#     for i, s in enumerate(sigma_list[1:]):
#         gaussian_current = ndi.gaussian_filter(img, sigma=s)
#         dog_image_cube[..., i] = gaussian_previous - gaussian_current
#         gaussian_previous = gaussian_current

#     # normalization factor for consistency in DoG magnitude
#     sf = 1 / (sigma_ratio - 1)
#     dog_image_cube *= sf

#     return dog_image_cube, sigma_list

if __name__ == "__main__":
    import sys
    from core.lib.utils import ColorTerminalStream
    sys.stdout = ColorTerminalStream()

    from core.parameters import IMAGE_SCALE
    from core.parameters import MEASURED_TIME, UTC_TIME, LATITUDE, LONGITUDE
    from core.parameters import INPUT_DIR

    ref_filename = "0.25000s_2024-04-09_02h40m33s.fits"
    other_filename = "0.25000s_2024-04-09_02h42m31s.fits"

    # ref_filename = "0.00100s_2024-04-09_02h39m59s.fits"
    # other_filenames = ["0.00100s_2024-04-09_02h43m02s.fits"]

    main(INPUT_DIR,
         ref_filename,
         other_filename)
    
    ### Legacy debug code below

    # fig0, ax0 = plt.subplots()
    # ax0.imshow(gaussian_filter((ref_img - registered_img)**2, sigma=10))

    # fig1, ax1 = plt.subplots()
    # anim_img = ax1.imshow(ref_img, animated=True)

    # def update(frame):
    #     if frame % 2 == 0:
    #         anim_img.set_data(ref_img)
    #     else:
    #         anim_img.set_data(registered_img)
    #     return [anim_img]

    # ani = FuncAnimation(fig1, update, frames=30, interval=500, blit=True)

    # # Cross correlation image
    # fig2, ax2 = plt.subplots()
    # # Centering
    # h, w = correlation_img.shape
    # extent = [-w//2, w//2-1, h//2 - 1, -h//2]
    # ax2.imshow(np.fft.fftshift(correlation_img), extent=extent)
    # # Zoom
    # r = 4
    # rolled_img = np.roll(np.roll(correlation_img, -ty+r, axis=0), -tx+r, axis=1)[:2*r+1, :2*r+1]
    # ax.imshow(rolled_img, extent=[tx-r, tx+r, ty+r, ty-r])

    # # Grid search registration
    # num = 5
    # theta_range = np.linspace(-np.deg2rad(0.062), -np.deg2rad(0.067), num)
    # tx_range = np.linspace(tx+0.16, tx+0.21, num)
    # ty_range = np.linspace(ty+0.26, ty+0.31, num)

    # values = np.zeros((num, num, num))
    # for i, theta in enumerate(theta_range):
    #     for j, tx in enumerate(tx_range):
    #         for k, ty in enumerate(ty_range):
    #             values[i,j,k] = obj.value(obj.convert_params_to_x(theta, tx, ty))

    # fig3, ax3 = plt.subplots(subplot_kw={"projection": "3d"})

    # vmin, vmax = values.min(), values.max()
    # X, Y = np.meshgrid(tx_range, ty_range, indexing='ij')

    # def update_surf(frame):
    #     theta_idx = frame % num
    #     tx_idx, ty_idx = np.unravel_index(np.argmin(values[theta_idx]), values[theta_idx].shape)
    #     tx, ty = tx_range[tx_idx], ty_range[ty_idx]
    #     value = values[theta_idx, tx_idx, ty_idx]
    #     ax3.cla()
    #     ax3.plot_surface(X, Y, values[theta_idx], edgecolor='0.7', alpha=0.8)
    #     ax3.scatter(tx, ty, value, c='red', label=f"theta: {np.rad2deg(theta_range[theta_idx]):.3f} \ntx: {tx:.2f} \nty: {ty:.2f} \nvalue: {value:.2e}")
    #     ax3.set_zlim(vmin, vmax)
    #     ax3.legend()

    #     return [fig3]
    
    # update_surf(0)
    # surf_ani = FuncAnimation(fig3, update_surf, frames=30, interval=500)
    
    # plt.show()