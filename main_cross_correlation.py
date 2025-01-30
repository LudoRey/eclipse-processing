import os
from matplotlib import pyplot as plt
import numpy as np
from skimage import transform
from matplotlib.animation import FuncAnimation

from lib.registration import centered_rigid_transform, line_search_gradient_descent, correlation, prep_for_registration, get_moon_clipping_value, DiscreteRigidRegistrationObjective
from lib.fits import read_fits_as_float, save_as_fits
from lib.display import center_crop
from lib.filters import gaussian_filter


def main(input_dir, ref_filename, other_filenames):
    
    # Load reference image
    ref_img, ref_header = read_fits_as_float(os.path.join(input_dir, ref_filename))
    h,w = ref_img.shape[0:2]
    # # TODO Center on the moon and crop 
    # moon_x, moon_y = ref_header["MOON-X"], ref_header["MOON-Y"]
    # w, h = 1024, 1024
    # ref_img, ref_header = center_crop(ref_img, int(moon_x), int(moon_y), w, h, ref_header)
    # Get clipping value (for all images)
    clipping_value = get_moon_clipping_value(ref_img, ref_header) # Possible bug : dead pixels
    # Prepare image for registration
    ref_img = prep_for_registration(ref_img, ref_header, clipping_value)

    for filename in other_filenames:
        # Load image
        img, header = read_fits_as_float(os.path.join(input_dir, filename))
        # # TODO Center on the moon and crop 
        # moon_x, moon_y = header["MOON-X"], header["MOON-Y"]
        # w, h = 1024, 1024
        # img, header = center_crop(img, int(moon_x), int(moon_y), w, h, header)
        # Prepare image for registration
        img = prep_for_registration(img, header, clipping_value)

        # Compute cross-correlation between img and ref_img
        # The highest peak minimizes the MSE w.r.t. translation ref_img -> img
        correlation_img = correlation(img, ref_img)
        ty, tx = np.unravel_index(np.argmax(correlation_img), correlation_img.shape)
        ty = ty if ty <= h // 2 else ty - h # ty in [0,h-1] -> [-h//2+1, h//2]
        tx = tx if tx <= w // 2 else tx - w
        theta = 0
        print("Coarse parameters:", np.rad2deg(theta), tx, ty)

        # We use it as an initial guess for the optimization-based approach
        obj = DiscreteRigidRegistrationObjective(ref_img, img)
        x0 = obj.convert_params_to_x(theta, tx, ty)

        x = line_search_gradient_descent(x0, obj.value, obj.grad)
        theta, tx, ty = obj.convert_x_to_params(x)

        print("Final parameters:", np.rad2deg(theta), tx, ty)
        
        tform = centered_rigid_transform(center=(w/2,h/2), rotation=theta, translation=(tx,ty))
        registered_img = transform.warp(img, tform) # inverse of the inverse here, be careful

        # max_index = np.unravel_index(np.argmax((registered_img-ref_img)**2), ref_img.shape)
        # print(max_index)

    fig0, ax0 = plt.subplots()
    ax0.imshow(gaussian_filter((ref_img - registered_img)**2, sigma=10))

    fig1, ax1 = plt.subplots()
    anim_img = ax1.imshow(ref_img, animated=True)

    def update(frame):
        if frame % 2 == 0:
            anim_img.set_data(ref_img)
        else:
            anim_img.set_data(registered_img)
        return [anim_img]

    ani = FuncAnimation(fig1, update, frames=30, interval=500, blit=True)

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

    # Grid search registration
    num = 5
    theta_range = np.linspace(-np.deg2rad(0.062), -np.deg2rad(0.067), num)
    tx_range = np.linspace(tx+0.16, tx+0.21, num)
    ty_range = np.linspace(ty+0.26, ty+0.31, num)

    values = np.zeros((num, num, num))
    # for i, theta in enumerate(theta_range):
    #     for j, tx in enumerate(tx_range):
    #         for k, ty in enumerate(ty_range):
    #             values[i,j,k] = obj.value(obj.convert_params_to_x(theta, tx, ty))

    fig3, ax3 = plt.subplots(subplot_kw={"projection": "3d"})

    vmin, vmax = values.min(), values.max()
    X, Y = np.meshgrid(tx_range, ty_range, indexing='ij')

    def update_surf(frame):
        theta_idx = frame % num
        tx_idx, ty_idx = np.unravel_index(np.argmin(values[theta_idx]), values[theta_idx].shape)
        tx, ty = tx_range[tx_idx], ty_range[ty_idx]
        value = values[theta_idx, tx_idx, ty_idx]
        ax3.cla()
        ax3.plot_surface(X, Y, values[theta_idx], edgecolor='0.7', alpha=0.8)
        ax3.scatter(tx, ty, value, c='red', label=f"theta: {np.rad2deg(theta_range[theta_idx]):.3f} \ntx: {tx:.2f} \nty: {ty:.2f} \nvalue: {value:.2e}")
        ax3.set_zlim(vmin, vmax)
        ax3.legend()

        return [fig3]
    
    update_surf(0)
    surf_ani = FuncAnimation(fig3, update_surf, frames=30, interval=500)

    anim_running = True
    def pause_anim(anim):
        nonlocal anim_running
        if anim_running:
            anim.event_source.stop()  # Stop the animation
            anim_running = False  # Set the flag to False, indicating the animation is paused
        else:
            anim.event_source.start()  # Start the animation
            anim_running = True  # Set the flag to True, indicating the animation is running

    fig3.canvas.mpl_connect('key_press_event', lambda event: pause_anim(surf_ani) if event.key == 'p' else None)
    
    plt.show()

if __name__ == "__main__":

    from parameters import IMAGE_SCALE
    from parameters import MEASURED_TIME, UTC_TIME, LATITUDE, LONGITUDE
    from parameters import INPUT_DIR

    ref_filename = "0.25000s_2024-04-09_02h40m33s.fits"
    other_filenames = ["0.25000s_2024-04-09_02h42m31s.fits"]

    # ref_filename = "0.00100s_2024-04-09_02h39m59s.fits"
    # other_filenames = ["0.00100s_2024-04-09_02h43m02s.fits"]

    main(INPUT_DIR,
         ref_filename,
         other_filenames)