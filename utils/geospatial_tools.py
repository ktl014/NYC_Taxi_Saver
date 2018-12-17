"""Geospatial Distance Calculation"""
# Standard dist imports
import numpy as np

# Module Level Constants
R = 6378    # Radius of the earth in kilometers


def haversine_np(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)

    All args must be of equal length.

    source: https://stackoverflow.com/a/29546836

    """
    # Convert latitude and longitude to radians
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    # Find the differences
    dlon = lon2 - lon1
    dlat = lat2 - lat1

    # Apply the formula
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(
        dlon / 2.0) ** 2
    # Calculate the angle (in radians)
    c = 2 * np.arcsin(np.sqrt(a))
    # Convert to kilometers
    km = R * c

    return km

def minkowski_distance(x1, x2, y1, y2, p):
    # p --> norm value (1 or 2)
    return ((abs(x2 - x1) ** p) + (abs(y2 - y1)) ** p) ** (1 / p)