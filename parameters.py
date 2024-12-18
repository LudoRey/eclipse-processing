from astropy.time import Time

# Platesolved parameters
IMAGE_SCALE = 5.89 # image scale in arcsec/pixels
ROTATION = 44.042 # given by PixInsight !!

# Time and location
MEASURED_TIME = '2024-04-09 02:39:56' # Measured time (doesnt need to be super precise)
UTC_TIME = '2024-04-08 19:29:25' # Known time (UTC !!)

LATITUDE = 45.93664
LONGITUDE = -70.48483

# Misc.
MOON_RADIUS_DEGREE = 0.278 # Stellarium gives 0.280 but its a bit too big. 0.278 is already an upper bound in practice.

GROUP_KEYWORDS = ["EXPTIME"]
#GROUP_KEYWORDS = ["EXPTIME", "ISOSPEED"] 

# I/O
INPUT_DIR = "D:\\_ECLIPSE2024\\data\\totality\\fits" # must be an existing directory : the others below will be created by the scripts.
MOON_DIR = "D:\\_ECLIPSE2024\\data\\totality\\moon_fits"
MOON_STACKS_DIR = "D:\\_ECLIPSE2024\\data\\totality\\moon_stacks"
SUN_DIR = "D:\\_ECLIPSE2024\\data\\totality\\sun_fits"
SUN_STACKS_DIR = "D:\\_ECLIPSE2024\\data\\totality\\sun_stacks"
SUN_HDR_DIR = "D:\\_ECLIPSE2024\\data\\totality\\sun_hdr"
MOON_HDR_DIR = "D:\\_ECLIPSE2024\\data\\totality\\moon_hdr"
MERGED_HDR_DIR = "D:\\_ECLIPSE2024\\data\\totality\\merged_hdr"