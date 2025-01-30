import numpy as np
from scipy import ndimage as ndi

def correlation(img1, img2):
    img1 = np.fft.fft2(img1) # Spectrum of image 1
    img2 = np.fft.fft2(img2) # Spectrum of image 2
    img = (img1 * np.conj(img2)) # Correlation spectrum
    img = np.real(np.fft.ifft2(img)) # Correlation image
    return img

def get_dog_cube(img, min_sigma=0.5, max_sigma=2, sigma_ratio=1.6):
    # k such that min_sigma*(sigma_ratio**k) > max_sigma
    k = int(np.mean(np.log(max_sigma / min_sigma) / np.log(sigma_ratio) + 1))

    # a geometric progression of standard deviations for gaussian kernels
    sigma_list = np.array([min_sigma * (sigma_ratio**i) for i in range(k + 1)])

    # computing difference between two successive Gaussian blurred images
    # to obtain an approximation of the scale invariant Laplacian of the
    # Gaussian operator
    dog_image_cube = np.empty(img.shape + (k,))
    gaussian_previous = ndi.gaussian_filter(img, sigma=sigma_list[0])
    for i, s in enumerate(sigma_list[1:]):
        gaussian_current = ndi.gaussian_filter(img, sigma=s)
        dog_image_cube[..., i] = gaussian_previous - gaussian_current
        gaussian_previous = gaussian_current

    # normalization factor for consistency in DoG magnitude
    sf = 1 / (sigma_ratio - 1)
    dog_image_cube *= sf

    return dog_image_cube, sigma_list