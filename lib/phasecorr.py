import numpy as np

from .disk import binary_disk
from .polar import warp_cart_to_polar, warp_polar_to_cart
from .filters import tangential_filter, gaussian_filter, partial_filter

from skimage import morphology

def prep_for_fft(img, header, saturation_value=0.11):
    print("Preparing image...")
    # Convert to grayscale
    if len(img.shape) == 3:
        img = img.mean(axis=2)

    moon_x, moon_y, moon_radius = header["MOON-X"], header["MOON-Y"], header["MOON-R"]
    # Mask out the moon and saturated pixels (set to a constant saturation_value)
    moon_mask = binary_disk(moon_x, moon_y, moon_radius*1.05, img.shape)
    saturation_mask = img >= saturation_value
    mask = saturation_mask | moon_mask
    img[mask] = saturation_value
    # High-pass filter
    img = img - tangential_filter(img, moon_x, moon_y, sigma=10)
    # Window to attenuate border discontinuities
    window = np.outer(np.hanning(img.shape[0]), np.hanning(img.shape[1])).reshape(img.shape)
    img = img*window

    return img

def prep_for_registration(img, header, clipping_value):
    print("Preparing image for registration...")
    # Convert to grayscale
    if len(img.shape) == 3:
        img = img.mean(axis=2)

    # Clip the moon and its surroundings
    moon_mask = binary_disk(header["MOON-X"], header["MOON-Y"], header["MOON-R"]*1.05, img.shape)
    clipping_mask = img >= clipping_value # should surround the moon_mask
    mask = clipping_mask | moon_mask
    img[mask] = clipping_value
    # High-pass filter
    img = img - tangential_filter(img, header["MOON-X"], header["MOON-Y"], sigma=10)
    return img # np.stack([mask, moon_mask, ext_moon_mask], axis=2, dtype=float)

def get_moon_clipping_value(img, header):
    # Convert to grayscale
    if len(img.shape) == 3:
        img = img.mean(axis=2)
    # Find clipping value that surrounds the 1.05R moon masks
    # The moon moves by less than 0.1R (~0.05R) during the eclipse : hence all moon masks will be contained by ext_moon_mask
    moon_mask = binary_disk(header["MOON-X"], header["MOON-Y"], header["MOON-R"]*1.05, img.shape) 
    ext_moon_mask = binary_disk(header["MOON-X"], header["MOON-Y"], header["MOON-R"]*1.15, img.shape) 
    moon_mask_border = ext_moon_mask & ~moon_mask
    clipping_value = np.min(img[moon_mask_border])
    return clipping_value 

def correlation(img1, img2):
    img1 = np.fft.fft2(img1) # Spectrum of image 1
    img2 = np.fft.fft2(img2) # Spectrum of image 2
    img = (img1 * np.conj(img2)) # Correlation spectrum
    img = np.real(np.fft.ifft2(img)) # Correlation image
    return img

def phase_correlation(img1, img2):
    img1 = np.fft.fft2(img1) # Spectrum of image 1
    img2 = np.fft.fft2(img2) # Spectrum of image 2
    img = (img1 * np.conj(img2)) / np.abs(img1 * np.conj(img2)) # Normalized correlation spectrum
    img = np.real(np.fft.ifft2(img)) # Normalized correlation image
    return img

def fft_logmag(img):
    print("Computing FFT...")
    fft = np.fft.fft2(img, axes=[0,1])
    fft_mag = np.abs(fft)
    fft_mag = np.fft.fftshift(fft_mag)
    return np.log1p(fft_mag)

def highpass_filter(spectrum, frequency):
    M, N = spectrum.shape 
    u = np.linspace(-0.5, 0.5, M)
    v = np.linspace(-0.5, 0.5, N)
    U, V = np.meshgrid(u, v)

    D2 = U**2 + V**2
    H = 1 - np.exp(-D2 / (2*frequency**2))
    return H*spectrum

def polar_transform(img):
    # Polar transform centered in the middle of the image
    y_c, x_c = img.shape[0] // 2, img.shape[1] // 2
    output_shape = [3600, 3600]
    polar_img, theta_factor, rho_factor = warp_cart_to_polar(img, x_c, y_c, output_shape, return_factors=True, log_scaling=False)
    # Inscribed circle only (warp_cart_to_polar returns circumscribed)
    border_rho = min(img.shape[0] - y_c, img.shape[1] - x_c) 
    polar_img = polar_img[:,:int(np.ceil(border_rho*rho_factor))]
    return polar_img

def subpixel_maximum(values):
    # Initial guess
    x0 = np.argmax(values)
    # Quadratic fit
    x = x0 - (values[x0+1] - values[x0-1]) / (2*(values[x0+1] - 2*values[x0] + values[x0-1]))
    return x