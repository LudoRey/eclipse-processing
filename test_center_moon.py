import os
from lib.fits import read_fits_as_float, save_as_fits
from lib.display import center_crop, crop
from matplotlib import pyplot as plt

import numpy as np

def roll_with_zero_padding(arr, shift, axis=0):
    arr = np.asarray(arr)
    result = np.zeros_like(arr)
    if shift == 0:
        return arr
    elif shift > 0:  # Shift right/down
        result_slice = [slice(None)] * arr.ndim
        arr_slice = [slice(None)] * arr.ndim
        result_slice[axis] = slice(shift, None)
        arr_slice[axis] = slice(None, -shift)
        result[tuple(result_slice)] = arr[tuple(arr_slice)]
    else:  # Shift left/up
        shift = -shift
        result_slice = [slice(None)] * arr.ndim
        arr_slice = [slice(None)] * arr.ndim
        result_slice[axis] = slice(None, -shift)
        arr_slice[axis] = slice(shift, None)
        result[tuple(result_slice)] = arr[tuple(arr_slice)]
    return result



def main(input_dir, output_dir, filenames):
    for filename in filenames:
        # Load image
        img, header = read_fits_as_float(os.path.join(input_dir, filename))
        h, w = img.shape[0:2]
        # Center on the moon
        int_moon_x, int_moon_y = int(header["MOON-X"]), int(header["MOON-Y"])
        img = roll_with_zero_padding(roll_with_zero_padding(img, h//2 - int_moon_y, axis=0), w//2 - int_moon_x, axis=1)
        save_as_fits(img, header, os.path.join(output_dir, filename))

if __name__ == "__main__":
    from parameters import INPUT_DIR
    output_dir = "D:\\_ECLIPSE2024\\data\\totality\\test"
    os.makedirs(output_dir)
    filenames = ["0.25000s_2024-04-09_02h40m33s.fits", "0.25000s_2024-04-09_02h42m31s.fits"]
    main(INPUT_DIR, output_dir, filenames)