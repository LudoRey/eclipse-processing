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

def auto_ht(x, clip_from_median=2.8, target_median=0.25, return_params=False):
    # Compute low clipping point
    median = np.median(x)
    MAD = np.median(np.abs(x-median))*1.4826 # constant to make consistent with gaussian distribution https://pixinsight.com/forum/index.php?threads/pixinsight-1-8-0-ripley-redesigned-statistics-tool.6119/
    low = median - clip_from_median*MAD
    # Compute high clipping point
    high = x.max()
    # Clip and rescale (usually done in ht() but we need it to compute m)
    x = np.clip(x, low, high)
    x = (x - low)/(high - low)
    # Compute the midpoint that yields a specified target median value
    median = np.median(x)
    m = mtf(median, target_median) # this might look weird but you can prove that it works
    # Apply histogram
    x = mtf(x, m)
    if return_params:
        return x, m, low, high
    else:
        return x

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
    # Open image/header
    with fits.open(filepath) as hdul:
        img = hdul[0].data
        header = hdul[0].header
    # Type checking and float conversion
    if np.issubdtype(img.dtype, np.uint16): 
        img = img.astype('float') / 65535
    elif np.issubdtype(img.dtype, np.floating):
        pass
    else:
        raise TypeError(f"FITS image format must be either 16-bit unsigned integer, or floating point.")
    # If color image : CxHxW -> HxWxC
    if len(img.shape) == 3:
        img = np.moveaxis(img, 0, 2)
    return img, header

def remove_pedestal(img, header):
    '''Updates header in-place'''
    if "PEDESTAL" in header:
        img = img - header["PEDESTAL"] / 65535
        img = np.maximum(img, 0)
        del header["PEDESTAL"]
    return img

def save_as_fits(img, header, filepath, convert_to_uint16=True):
    print(f"Saving to {filepath}...")
    if convert_to_uint16:
        img = (np.clip(img, 0, 1)*65535).astype('uint16')
    if len(img.shape) == 3:
        img = np.moveaxis(img, 2, 0)
    hdu = fits.PrimaryHDU(data=img, header=header)
    hdu.writeto(filepath, overwrite=True)

def read_fits_header(filepath):
    with fits.open(filepath) as hdul:
        header = hdul[0].header
    return header

def extract_subheader(header, keys):
    kv_dict = {}
    for k in keys:
        if k in header.keys():
            kv_dict[k] = header[k]
    subheader = fits.Header(kv_dict)
    return subheader

def combine_headers(header1, header2):
    # common keys will be overriden by header2's keywords
    kv_dict = {}
    for k in header1.keys():
        kv_dict[k] = header1[k]
    for k in header2.keys():
        kv_dict[k] = header2[k]
    header = fits.Header(kv_dict)
    return header

def get_grouped_filepaths(dirname, keywords, output_format="collapsed_dict"):
    # Based on an ordered list of keywords, group into a nested dict, then sort into an ordered dict, then collapse the nested_dict
    # Example : keywords = ["EXPTIME", "ISOSPEED"]
    # Output format options {"nested_dict", "collapsed_dict", "collapsed_list"}
    # Nested dict structure : {"0.25": {"100": ["a", "b"]}, "1": {"100": ["c"], "200": ["d"]}}
    # Collapsed dict structure : {"0.25-100": ["a", "b"], "1-100": ["c"], "1-200": ["d"]}
    # Collapsed list structure : [["a", "b"], ["c"], ["d"]]
    nested_dict = {}
    dirpath, _, filenames = next(os.walk(dirname)) # not going into subfolders
    for filename in filenames:
        if filename.endswith('.fits'):
            filepath = os.path.join(dirpath, filename)
            header = read_fits_header(filepath)
            # Create/access deepest level of nested dict
            sub_dict = nested_dict
            for keyword in keywords[:-1]:
                key = str(header[keyword])
                if key not in sub_dict.keys():
                    sub_dict[key] = {}
                sub_dict = sub_dict[key]
            # Deepest level : sub_dict is a simple dict with filepaths lists as values, and last keyword values as keys (e.g. the ISO value)
            keyword = keywords[-1]
            key = str(header[keyword])
            if key in sub_dict.keys():
                sub_dict[key].append(filepath)
            else:
                sub_dict[key] = [filepath]
    # Sort nested dict
    nested_dict = sort_nested_dict(nested_dict)
    # Collapse dict
    collapsed_dict = collapse_nested_dict(nested_dict, keywords)
    if output_format == "nested_dict":
        return nested_dict
    elif output_format == "collapsed_dict":
        return collapsed_dict
    elif output_format == "collapsed_list":
        return [collapsed_dict[k] for k in collapsed_dict.keys()]
    else:
        return ValueError('output_format must be one of {"nested_dict", "collapsed_dict", "collapsed_list"}')
        

def sort_nested_dict(nested_dict, cvt_key_to_float=False):
    # Sort by lexicographical order by default, where "10" < "2". This is usually not the expected behavior when keys are float.
    for key in nested_dict.keys():
        if isinstance(nested_dict[key], dict):
            nested_dict[key] = sort_nested_dict(nested_dict[key], cvt_key_to_float)
    if cvt_key_to_float:
        keymap = lambda kv: float(kv[0])
    else:
        keymap = lambda kv: kv[0]
    nested_dict = dict(sorted(nested_dict.items(), key=keymap))
    return nested_dict

def collapse_nested_dict(nested_dict, keywords):
    collapsed_dict = {}
    for key in nested_dict.keys():
        if isinstance(nested_dict[key], dict):
            nested_dict[key] = collapse_nested_dict(nested_dict[key], keywords[1:])
            for subkey in nested_dict[key].keys():
                collapsed_dict[format_keyword_str(key, keywords[0])+"_"+subkey] = nested_dict[key][subkey]
        else: # in that case, nested_dict is a regular dict and only the keys need to be formatted
            collapsed_dict[format_keyword_str(key, keywords[0])] = nested_dict[key]
    return collapsed_dict

def format_keyword_str(str_value, keyword):
    if keyword == "EXPTIME":
        return f"EXP-{float(str_value):.5f}"
    elif keyword == "ISOSPEED" or keyword == "GAIN":
        return f"GAIN-{int(float(str_value))}"
    else:
        return str_value

def crop_inset(img, crop_center, crop_radii, scale=4, border_value=np.nan, border_thickness=2):
    # Crop
    i_left, i_right = crop_center[0]-crop_radii[0], crop_center[0]+crop_radii[0]
    j_top, j_bottom = crop_center[1]-crop_radii[1], crop_center[1]+crop_radii[1]
    crop = img[i_left:i_right+1, j_top:j_bottom+1]
    # Crop border
    img[i_left-border_thickness:i_right+1+border_thickness, j_top-border_thickness:j_top] = border_value
    img[i_left-border_thickness:i_right+1+border_thickness, j_bottom+1:j_bottom+1+border_thickness] = border_value
    img[i_left-border_thickness:i_left, j_top-border_thickness:j_bottom+1+border_thickness] = border_value
    img[i_right+1:i_right+1+border_thickness, j_top-border_thickness:j_bottom+1+border_thickness] = border_value
    # Add inset
    inset = crop.repeat(scale,axis=0).repeat(scale,axis=1)
    img[-inset.shape[0]:, -inset.shape[1]:] = inset
    # Inset border 
    img[-inset.shape[0]:, -inset.shape[1]-border_thickness:-inset.shape[1]] = border_value
    img[-inset.shape[0]-border_thickness:-inset.shape[0], -inset.shape[1]:] = border_value

def crop_img(img, left, right, top, bottom, header=None):
    # Crop image
    new_img = img[top:bottom+1, left:right+1]
    if header is not None:
        # Create and update new header
        new_header = fits.Header(header, copy=True)
        new_header["NAXIS1"], new_header["NAXIS2"] = img.shape[1], img.shape[0] 
        for k, v in new_header.items():
            if k in ["MOON-X", "SUN-X", "TRANS-X"]:
                new_header[k] = v - left 
            if k in ["MOON-Y", "SUN-Y", "TRANS-Y"]:
                new_header[k] = v - top
        return new_img, new_header 
    else:
        return new_img