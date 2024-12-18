# Eclipse processing guide

## Requirements

- Python 3.7+
- Packages in `requirements.txt`

## Parameters

The parameters in `parameters.py` should be set once and for all : they are not tuneable parameters.
- Platesolved parameters. The rotation is required for the sun alignment algorithm (see below).
    - `IMAGE_SCALE` : image scale given in arcsec/pixel.
    - `ROTATION` : rotation angle in degrees (modulo 360) returned by a plate solving software. This is the [position angle](https://en.wikipedia.org/wiki/Position_angle) of a vector pointing upward in your images. **Warning:** if platesolving with [nova.astrometry.net](nova.astrometry.net), add 180 degrees to the result and make sure to use FITS files as input (PNG or JPEG files might result in flipped angles). To double check the result, display the RA/DEC grid and make sure that the angle matches the position angle convention. 
- Time and location. The sun alignment algorithm is based on an ephemeris table (like Stellarium, but less fun) that requires the time and location of observation.
    - `MEASURED_TIME`, `UTC_TIME` : The camera time may not be equal to UTC time, and even if we account for timezones, the camera time could simply be wrong. To account for this, we need to compute an offset. In order to do so, we need to consider an event which 1. was recorded by the camera and 2. occured at a known time. This event can be C2 or C3 for example, for which the times can be found [here](http://xjubier.free.fr/en/site_pages/solar_eclipses/TSE_2024_GoogleMapFull.html). Both UTC and measured timestamps should be provided in YYYY-MM-DD HH:MM:SS format.
    - `LATITUDE`, `LONGITUDE` : latitude and longitude in decimal degrees. These can also be found on Xavier Jubier's interactive map.
- Miscellaneous.
    - `MOON_RADIUS_DEGREE` : Radius of the moon in degrees (specific to a certain TSE). This value can be found in Stellarium or derived from your own images (Stellarium tends to overestimate it). For the 2024 TSE, 0.278 is a good value.
    - `GROUP_KEYWORDS` : List of FITS keywords corresponding to settings that vary across the exposures (typically, "EXPTIME" and optionally "ISOSPEED" or "GAIN" if the gain was changed). These keywords will automatically determine groups of images to be stacked together. The keywords should be listed by order of importance : groups will be sorted by brightness based on the first keyword in priority, then on the second, etc... The order is important for the HDR algorithm.
- I/O.
    - `INPUT_DIR` : Input directory for the registration script (`main_registration.py`) containing the calibrated (and debayered) images of the TSE in 16-bit unsigned integer FITS format (.fits extension, can use PixInsight's BatchFormatConversion script). **Do not** sort your images into subfolders based on exposure time or gain. The scripts will automatically detect the different settings based on `GROUP_KEYWORDS`.
    - `MOON_DIR`, `SUN_DIR`, `MOON_STACKS_DIR`, `SUN_STACKS_DIR`, `MOON_STACKS_DIR`, `SUN_STACKS_DIR`, `MOON_HDR_DIR`, `SUN_HDR_DIR`, `MERGED_HDR_DIR` : Output (and input) directories for the scripts.

Each script contains its own set of parameters, listed at the top of the file. More details are given in the sections below.

## Registration

The script `main_registration.py` simultaneously performs a moon-based and a sun-based registration of the input images located in `INPUT_DIR`. The output directories are defined by `MOON_DIR` and `SUN_DIR`.

Extra parameters (defined at the bottom of the script) :
- `REF_FILENAME` : The registration is based on a reference image located at <`INPUT_DIR`>/<`REF_FILENAME`>.

## Integration

The scripts `main_sun_integration.py` and `main_moon_integration.py` integrate the previously registered images located in `MOON_DIR` and `SUN_DIR`. A stack is generated for each group (see `GROUP_KEYWORDS`). The output directories are defined by `MOON_STACKS_DIR` and `SUN_STACKS_DIR`.

The `main_sun_integration.py` script performs a weighted average of each pixel in order to reject as many moon pixels as possible. For each sub, a moon mask is computed, which depends on two additional parameters : 
- `EXTRA_RADIUS_PIXELS` : extra amount of pixels added to the radius of the moon mask (which is obtained from `MOON_RADIUS_DEGREE` and `IMAGE_SCALE`). Increasing this parameter will lead to fewer artifacts at the cost of worse SNR : it should be as close to 0 as possible.
- `SMOOTHNESS` : smoothness of the mask in pixels. Increasing this parameter leads to a smoother transition at the cost of worse SNR.

## HDR composition

The scripts `main_sun_hdr_composition.py` and `main_moon_hdr_composition.py` combine the previously generated stacks located in `MOON_STACKS_DIR` and `SUN_STACKS_DIR`. The output directories are defined by `MOON_HDR_DIR` and `SUN_HDR_DIR`.

Because they are stored in 16-bit files, the pixel values of an image taken with a 14-bit sensor typically saturate at 0.25 (in the normalized [0,1] range), but this value can even be lower based on the full well capacity (FWC) of the sensor. Even then, the sensor might not be linear near the saturation point : values above ~80-90% of the saturation point are often not representative of the true brightness. Similarly, values that are near 0 suffer from the same issues. In order to create a smooth and realistic HDR composite, those too-bright and too-dark values should be rejected by the HDR algorithm. However, those thresholds uniquely depend on the imaging system, and should be derived from the images themselves. Be careful: image calibration (bias subtraction and flat division) has a non-uniform effect on those thresholds : some pixels might saturate at a lower/higher point than others for example. It is usually best to reject more pixels than necessary (as opposed to not enough).

In essence, the HDR algorithm performs a weighted combination, where the pixels that are too bright (or too dark) are rejected based on a weighting function defined by 4 parameters :
- `HIGH_CLIPPING_THRESHOLD`, `HIGH_CLIPPING_SMOOTHNESS` : values in [0,1]. The weight function is equal to 1 for pixel values below `HIGH_CLIPPING_THRESHOLD`, and equal to 0 above `HIGH_CLIPPING_THRESHOLD`+`HIGH_CLIPPING_SMOOTHNESS`. Between the two, it is a simple linear interpolation. 
- `LOW_CLIPPING_THRESHOLD`, `LOW_CLIPPING_SMOOTHNESS` : analogous to `HIGH_CLIPPING_THRESHOLD` and `HIGH_CLIPPING_SMOOTHNESS`.

Moreover, `main_sun_hdr_composition.py` uses a fitting routine before combining the images. The fit is computed on a region of appropriate brightness (as defined by `HIGH_CLIPPING_THRESHOLD` and `LOW_CLIPPING_THRESHOLD`), which also excludes the moon. Similarly to `main_sun_integration.py`, the script uses an additional parameter for the moon mask :
- `EXTRA_RADIUS_PIXELS` : extra amount of pixels added to the radius of the moon mask (which is obtained from `MOON_RADIUS_DEGREE` and `IMAGE_SCALE`).

## Moon and sun composition 

The script `main_merge_sun_moon.py` combines the previously generated HDR images located in `MOON_HDR_DIR` and `SUN_HDR_DIR`. The output directory is defined by `MERGED_HDR_DIR`. 

The script uses a moon mask (once again!), but this time it is not approximated by a disk but rather directly estimated from the image. 
- `MOON_THRESHOLD` : value in [0,1]. Only moon pixels below this value will be considered for the initial moon mask. This value should be increased to contain more of the moon edge, but it should not be too high (to avoid artifacts). 
- `SIGMA` : value above 0. Roughly corresponds to "outwards-only" Gaussian smoothing (but there is more to it, more explanations will come later).