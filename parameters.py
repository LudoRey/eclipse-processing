from astropy.time import Time

# Platesolved parameters
IMAGE_SCALE = 5.89 # image scale in arcsec/pixels

ROTATION = 44.045 # given by PixInsight !!

# Observation parameters
LATITUDE = 45.93664
LONGITUDE = -70.48483
# Camera time may not be equal to UTC time, so we need to compute an offset !
# Even if we account for timezones, the camera time could simply be wrong.
# Reference time can be C2 or C3, or even derived after shooting (as long as the camera time was not reset)
# http://xjubier.free.fr/en/site_pages/solar_eclipses/TSE_2024_GoogleMapFull.html
ref_time_utc = '2024-04-08 19:29:25' # UTC !!
ref_time_measured = '2024-04-09 02:39:56' # Doesnt need to be super precise, since only the angle of the displacement wrt ref is relevant

TIME_OFFSET = Time(ref_time_measured, scale='utc') - Time(ref_time_utc, scale='utc')

# Radius of the moon in degrees (approximately)
MOON_RADIUS_DEGREE = 0.278 # Stellarium gives 0.280 but its a bit too big. 0.278 is already an upper bound in practice.

# Derived manually from images
SATURATION_VALUE = 0.13
CLIP_EXP_TIME = 0.02 # fastest exposure time with clipping around the moon

# I/O
INPUT_DIR = "data\\totality\\fits"
MOON_DIR = "data\\totality\\moon_fits"
MOON_STACKS_DIR = "data\\totality\\moon_stacks"
SUN_DIR = "data\\totality\\sun_fits"
SUN_STACKS_DIR = "data\\totality\\sun_stacks"

REF_FILENAME = "0m00025s_2024-04-09_02h42m41s.fits"
FILENAME = "0m00025s_2024-04-09_02h40m16s.fits"
#REF_FILENAME = "0m00025s_2024-04-09_02h43m02s.fits"
#FILENAME = "0m00025s_2024-04-09_02h39m59s.fits"