import numpy as np
import skimage as sk

from core.lib.disk import binary_disk
from core.lib import filters, transform, optim

def get_clipping_value(img, header):
    # Convert to grayscale
    if len(img.shape) == 3:
        img = img.min(axis=2)
    # Find clipping value that surrounds the 1.05R moon mask
    moon_mask = binary_disk(header["MOON-X"], header["MOON-Y"], header["MOON-R"]*1.05, img.shape)
    moon_mask_border = sk.morphology.binary_dilation(moon_mask) & ~moon_mask
    clipping_value = np.min(img[moon_mask_border]) # Possible bug : dead pixels
    return clipping_value

def preprocess(img, header, clipping_value, sigma_high_pass_tangential=10, sigma_low_pass=3):
    print("Preprocessing image...", end=" ", flush=True)
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

    # Tangential high-pass filter
    img = img - filters.tangential_filter(img, (header["MOON-X"], header["MOON-Y"]), sigma=sigma_high_pass_tangential)
    # Low-pass filter to match the bilinear interpolation smoothing that happens during registration
    img = filters.gaussian_filter(img, sigma=sigma_low_pass)
    
    img /= img.std()
    print("Done.")
    return img


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