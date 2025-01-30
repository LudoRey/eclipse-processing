import numpy as np
from astropy.time import Time
from astropy.coordinates import EarthLocation, get_body

# Old astrometry approach
def get_moon_radius(time: Time, location: EarthLocation):
    moon_coords = get_body("moon", time, location)
    earth_coords = get_body("earth", time, location)
    moon_dist_km = earth_coords.separation_3d(moon_coords).km
    moon_real_radius_km = 1737.4
    moon_radius_degree = np.arctan(moon_real_radius_km / moon_dist_km) * 180 / np.pi
    return moon_radius_degree

def get_sun_moon_offset(time: Time, location: EarthLocation):
    # Get the moon and sun coordinates at the specified time and location
    moon_coords = get_body("moon", time, location)
    sun_coords = get_body("sun", time, location)
    # Compute the offset
    sun_offset_scalar = moon_coords.separation(sun_coords).arcsecond
    sun_offset_angle = moon_coords.position_angle(sun_coords).degree
    return sun_offset_scalar, sun_offset_angle

def convert_angular_offset_to_x_y(offset_scalar, offset_angle, camera_rotation, image_scale):
    '''
    offset_scalar is given in arcseconds
    offset_angle and camera_rotation are given in degrees
    image_scale is given in arcseconds/pixel
    '''
    # 1) offset is offset_angle degrees east of north
    # 2) up (-y) is camera_rotation degrees east of north
    # Combining 1) and 2), offset is offset_angle + (360 - camera_rotation) degrees counterclockwise of up
    # Modulo 360, offset is offset_angle - camera_rotation + 90 degrees counterclockwise of x
    offset_angle_to_x = (offset_angle - camera_rotation + 90) * np.pi / 180 # counterclockwise
    offset_x = np.cos(offset_angle_to_x)*offset_scalar / image_scale
    offset_y = -np.sin(offset_angle_to_x)*offset_scalar / image_scale
    return offset_x, offset_y