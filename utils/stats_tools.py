""" """
# Standard dist imports

# Third party imports
import numpy as np

# Project level imports

# Module level constants

def ecdf(x):
    """Empirical cumulative distribution function of a variable"""
    # Sort in ascending order
    x = np.sort(x)
    n = len(x)

    # Go from 1/n to 1
    y = np.arange(1, n + 1, 1) / n

    return x, y