import numpy as np
import skimage as sk
from scipy import ndimage as ndi

from lib.disk import binary_disk
from lib.filters import tangential_filter
from lib.registration import optim, utils

def get_clipping_value(img, header):
    # Convert to grayscale
    if len(img.shape) == 3:
        img = img.min(axis=2)
    # Find clipping value that surrounds the 1.05R moon masks
    # The moon moves by less than 0.1R (~0.05R) during the eclipse : hence all moon masks will be contained by ext_moon_mask
    moon_mask = binary_disk(header["MOON-X"], header["MOON-Y"], header["MOON-R"]*1.05, img.shape) 
    ext_moon_mask = binary_disk(header["MOON-X"], header["MOON-Y"], header["MOON-R"]*1.15, img.shape) 
    moon_mask_border = ext_moon_mask & ~moon_mask
    clipping_value = np.min(img[moon_mask_border])
    return clipping_value

def preprocess(img, header, clipping_value):
    print("Preparing image for registration...")
    # Convert to grayscale
    if len(img.shape) == 3:
        img = img.mean(axis=2)
    # Clip the moon and its surroundings
    moon_mask = binary_disk(header["MOON-X"], header["MOON-Y"], header["MOON-R"]*1.05, img.shape)
    clipping_mask = img >= clipping_value # should surround the moon_mask
    mask = clipping_mask | moon_mask
    img[mask] = clipping_value
    # Normalize
    img /= clipping_value

    # # Inpaint stars and hot pixels
    # print("Creating DoG cube")
    # dog_cube, sigma_list = utils.get_dog_cube(img, 0.5, 2)

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

    print("Bandpass filter")
    # High-pass tangential filter
    img = img - tangential_filter(img, header["MOON-X"], header["MOON-Y"], sigma=10)
    # Low pass filter (to match the bilinear interpolation smoothing that happends during registration)
    img = ndi.gaussian_filter(img, sigma=2)
    
    img /= img.std()
    return img

def register(img, ref_img, rotation_center):
    h, w = img.shape[0:2]
    # Compute cross-correlation between img and ref_img
    # The highest peak minimizes the MSE w.r.t. integer translation ref_img -> img
    correlation_img = utils.correlation(img, ref_img)
    ty, tx = np.unravel_index(np.argmax(correlation_img), correlation_img.shape)
    ty = ty if ty <= h // 2 else ty - h # ty in [0,h-1] -> [-h//2+1, h//2]
    tx = tx if tx <= w // 2 else tx - w
    theta = 0
    print("Coarse parameters:", np.rad2deg(theta), tx, ty)

    # We use it as an initial guess for the optimization-based approach
    obj = optim.DiscreteRigidRegistrationObjective(ref_img, img, rotation_center)
    x0 = obj.convert_params_to_x(theta, tx, ty)

    x = optim.line_search_gradient_descent(x0, obj.value, obj.grad)
    theta, tx, ty = obj.convert_x_to_params(x)

    print("Final parameters:", np.rad2deg(theta), tx, ty)
    return theta, tx, ty