import numpy as np
import os
from astropy.io import fits
import time

class Timer:
    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        elapsed_time = self.end_time - self.start_time
        print(f"Elapsed time: {elapsed_time:.2f} seconds")

def crop(img, x_c, y_c, w=512, h=512):
    return img[y_c-h//2:y_c+h//2, x_c-w//2:x_c+w//2]

def combine_red_green(img1, img2):
    img = np.zeros((img1.shape[0], img1.shape[1], 3))
    img[:,:,0] = img1
    img[:,:,1] = img2 
    return img

def ht(x, m, low=None, high=None):
    if low is None:
        low = x.min()
    if high is None:
        high = x.max()
    x = np.clip(x, low, high)
    x = (x - low)/(high - low)
    x = mtf(x, m)
    return x

def mtf(x, m):
    return (m-1)*x/((2*m-1)*x-m)

def read_fits_as_float(filepath, verbose=True):
    if verbose:
        print(f"Opening {filepath}...")
    with fits.open(filepath) as hdul:
        img = hdul[0].data
        header = hdul[0].header
    if len(img.shape) == 3:
        img = np.moveaxis(img, 0, 2) # CxHxW -> HxWxC
    img = img.astype('float') / 65535
    return img, header

def read_fits(filepath, verbose=True, to_float=False):
    if verbose:
        print(f"Opening {filepath}...")
    with fits.open(filepath) as hdul:
        img = hdul[0].data
        header = hdul[0].header
    if len(img.shape) == 3:
        img = np.moveaxis(img, 0, 2) # CxHxW -> HxWxC
    if to_float:
        img = img.astype('float') / 65535
    return img, header

def save_as_fits(img, header, filepath):
    print(f"Saving to {filepath}...")
    img = (np.clip(img, 0, 1)*65535).astype('uint16')
    data = np.moveaxis(img, 2, 0) # HxWxC -> CxHxW
    hdu = fits.PrimaryHDU(data=data, header=header)
    hdu.writeto(filepath, overwrite=True)

def save_as_float_fits(img, header, filepath):
    print(f"Saving to {filepath}...")
    data = np.moveaxis(img, 2, 0) # HxWxC -> CxHxW
    hdu = fits.PrimaryHDU(data=data, header=header)
    hdu.writeto(filepath, overwrite=True)

def read_fits_header(filepath):
    with fits.open(filepath) as hdul:
        header = hdul[0].header
    return header

def extract_subheader(header, keys):
    kv_dict = {}
    for k in keys:
        kv_dict[k] = header[k]
    subheader = fits.Header(kv_dict)
    return subheader

def get_filepaths_per_exptime(dirname):
    filepaths_per_exptime = {}
    dirpath, _, filenames = next(os.walk(dirname)) # not going into subfolders
    for filename in filenames:
        if filename.endswith('.fits'):
            filepath = os.path.join(dirpath, filename)
            header = read_fits_header(filepath)
            if str(header["EXPTIME"]) in filepaths_per_exptime.keys():
                filepaths_per_exptime[str(header["EXPTIME"])].append(filepath)
            else:
                filepaths_per_exptime[str(header["EXPTIME"])] = [filepath]
    return filepaths_per_exptime