import numpy as np
import os
from astropy.io import fits

def read_fits_as_float(filepath, rows_range=None, verbose=True):
    if verbose:
        print(f"Opening {filepath}...")
    # Open image/header
    with fits.open(filepath) as hdul:
        header = hdul[0].header
        if rows_range is None:
            img = hdul[0].data
        else:
            img = hdul[0].data[:,rows_range[0]:rows_range[1]]
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
                key = format_keyword_value(header[keyword], keyword)
                if key not in sub_dict.keys():
                    sub_dict[key] = {}
                sub_dict = sub_dict[key]
            # Deepest level : sub_dict is a simple dict with filepaths lists as values, and last keyword values as keys (e.g. the ISO value)
            keyword = keywords[-1]
            key = format_keyword_value(header[keyword], keyword)
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
    if output_format == "collapsed_dict":
        return collapsed_dict
    if output_format == "collapsed_list":
        return [collapsed_dict[k] for k in collapsed_dict.keys()]
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
            collapsed_subdict = collapse_nested_dict(nested_dict[key], keywords[1:])
            for subkey in collapsed_subdict.keys():
                collapsed_dict[(key,)+subkey] = collapsed_subdict[subkey]
        else: # in that case, nested_dict is a regular dict and only the keys need to be formatted
            collapsed_dict[(key,)] = nested_dict[key]
    return collapsed_dict

def format_keyword_value(keyword_value, keyword):
    if keyword == "EXPTIME":
        return f"{keyword_value:.5f}"
    elif keyword == "ISOSPEED" or keyword == "GAIN":
        return f"{int(keyword_value)}"
    else:
        return f"{keyword_value}"