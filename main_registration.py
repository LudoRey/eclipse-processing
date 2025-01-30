import os
import numpy as np
import skimage as sk
from scipy import ndimage as ndi
from matplotlib import pyplot as plt, animation as anim

from lib import fits, display
from lib.registration import sun, transform

def main(input_dir, ref_filename, other_filenames):
    # Load reference image
    ref_img, ref_header = fits.read_fits_as_float(os.path.join(input_dir, ref_filename))

    w, h = 1024, 1024
    #ref_img, ref_header = display.center_crop(ref_img, int(ref_header["MOON-X"]), int(ref_header["MOON-Y"]), w, h, ref_header)

    clipping_value = sun.get_clipping_value(ref_img, ref_header)
    ref_img = sun.preprocess(ref_img, ref_header, clipping_value)
    INTERFACE.imshow(ref_img)

    for filename in other_filenames:
        img, header = fits.read_fits_as_float(os.path.join(input_dir, filename))
        #img, header = display.center_crop(img, int(header["MOON-X"]), int(header["MOON-Y"]), w, h, header)
        img = sun.preprocess(img, header, clipping_value)
        theta, tx, ty = sun.register(img, ref_img, rotation_center=(header["MOON-X"], header["MOON-Y"]))
        print(np.rad2deg(theta), tx, ty)

        tform = transform.centered_rigid_transform(center=(header["MOON-X"], header["MOON-Y"]), rotation=theta, translation=(tx,ty))
        registered_img = sk.transform.warp(img, tform) # inverse of the inverse here, be careful

    fig, ax = plt.subplots()
    
    rgb = np.stack([ref_img, 0.5*(ref_img+registered_img), registered_img], axis=2)
    rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min())
    ax.imshow(rgb)
    #ax.imshow(ndi.gaussian_filter((ref_img - registered_img)**2, sigma=10))

    plt.show()

if __name__ == "__main__":

    from parameters import INPUT_DIR
    from lib.interface import DefaultInterface

    global INTERFACE
    INTERFACE = DefaultInterface()

    ref_filename = "0.25000s_2024-04-09_02h40m33s.fits"
    other_filenames = ["0.25000s_2024-04-09_02h42m31s.fits"]

    # ref_filename = "0.00100s_2024-04-09_02h39m59s.fits"
    # other_filenames = ["0.00100s_2024-04-09_02h43m02s.fits"]

    main(INPUT_DIR,
         ref_filename,
         other_filenames)