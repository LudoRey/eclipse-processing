from lib.registration import get_sun_moon_offset, convert_angular_offset_to_x_y

from matplotlib import pyplot as plt

from astropy.coordinates import EarthLocation, get_body, SkyCoord
from astropy.time import Time, TimeDelta
import astropy.units as u

import numpy as np

location = EarthLocation(lat=45.93664, lon=-70.48483, height=0)
time_start = Time('2024-04-08 19:29:25', scale='utc')
time_start += TimeDelta(np.random.random()*u.year)
print(time_start)
time_end = time_start + TimeDelta(2*u.minute)



sun_start = get_body("sun", time_start, location)
sun_end = get_body("sun", time_end, location)

sun_sep = sun_start.separation(sun_end)
print("Separation:", sun_sep.arcsec)

sun_start = SkyCoord(sun_start.ra, sun_start.dec)
sun_end = SkyCoord(sun_end.ra, sun_end.dec)

sun_sep = sun_start.separation(sun_end)
print("Separation:", sun_sep.arcsec)

print("RA:", sun_start.ra, "DEC:", sun_start.dec)
print("RA:", sun_end.ra, "DEC:", sun_end.dec)

# x, y = np.zeros(100), np.zeros(100)
# offsets = np.linspace(-1/24, 1/24, 100)
# times = [time + offset for offset in offsets]

# for i in range(100):
#     x[i], y[i] = convert_angular_offset_to_x_y(*get_sun_moon_offset(times[i], location), 0, 1)

# fig, axes = plt.subplots(2)
# axes[0].scatter(x, y, c=offsets)
# x_line = np.linspace(x[0], x[-1], 100)
# y_line = np.linspace(y[0], y[-1], 100)
# axes[1].plot(np.sqrt((x-x_line)**2 + (y-y_line)**2))
# plt.show()