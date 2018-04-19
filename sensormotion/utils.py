"""
Utility functions used across the rest of the package.
"""

from __future__ import division


def ceildiv(a, b):
    """Ceiling division.

    Divide and round up the result.

    Parameters
    ----------
    a : int or float
        dividend (numerator) to be divided.
    b : int or float
        divisor (denominator) to divide by.

    Returns
    -------
    result : float
        quotient of the division, rounded up to the nearest integer.
    """

    return -(-a // b)
