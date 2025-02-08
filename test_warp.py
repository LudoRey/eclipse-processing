import os 
import sys
# Since the script is located inside core, need to add project root to path in order to recognize core as a package
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.append(project_root)

import cv2 
import skimage as sk 
import numpy as np
from scipy import ndimage as ndi

from matplotlib import pyplot as plt

from core.lib.utils import Timer

from core.lib import polar, transform, filters

img1 = sk.data.camera()
img1 = img1.astype('float32') / 255.0
img1 = sk.transform.rescale(img1, 10)

img2 = img1

fig, axes = plt.subplots(2)

def np_correlation(img1, img2):
    img1 = np.fft.fft2(img1) # Spectrum of image 1
    img2 = np.fft.fft2(img2) # Spectrum of image 2
    img = (img1 * np.conj(img2)) # Correlation spectrum
    img = np.real(np.fft.ifft2(img)) # Correlation image
    return img

def cv_correlation(img1, img2):
    img1 = cv2.dft(img1, flags=cv2.DFT_COMPLEX_OUTPUT)
    img2 = cv2.dft(img2, flags=cv2.DFT_COMPLEX_OUTPUT)
    correlation_spectrum = cv2.mulSpectrums(img1, img2, 0, conjB=True)
    return cv2.idft(correlation_spectrum, flags=cv2.DFT_REAL_OUTPUT | cv2.DFT_SCALE) # Correlation image

with Timer():
    correlation_img = np_correlation(img1, img2)

axes[0].imshow(correlation_img)

with Timer():
    correlation_img = cv_correlation(img1, img2)

axes[1].imshow(correlation_img)

plt.show()