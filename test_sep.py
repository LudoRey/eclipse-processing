from lib.registration import get_sun_moon_offset, convert_angular_offset_to_x_y

from matplotlib import pyplot as plt

from astropy.coordinates import EarthLocation
from astropy.time import Time

import numpy as np

location = EarthLocation(lat=45.93664, lon=-70.48483, height=0)
time = Time('2024-04-08 19:29:25', scale='utc')

x, y = np.zeros(100), np.zeros(100)
offsets = np.linspace(-1/24, 1/24, 100)
times = [time + offset for offset in offsets]

for i in range(100):
    x[i], y[i] = convert_angular_offset_to_x_y(*get_sun_moon_offset(times[i], location), 0, 1)

fig, axes = plt.subplots(2)
axes[0].scatter(x, y, c=offsets)
x_line = np.linspace(x[0], x[-1], 100)
y_line = np.linspace(y[0], y[-1], 100)
axes[1].plot(np.sqrt((x-x_line)**2 + (y-y_line)**2))
plt.show()