"""
gaitdynamics
============

Provides tools for estimating gait dynamics from accelerometer data. Also
contains a few useful functions for pre-processing and visualizing
accelerometer signals.

This package was primarily developed for use on Android sensor data collected
at the Attentional Neuroscience Lab (University of British Columbia).

Documentation
-------------
Documentation is available via docstrings provided with the code, and an
online API reference found at
`ReadTheDocs <http://gaitdynamics.readthedocs.io>`_.

To view documentation for a function or module, first make sure the package
has been imported:

  >>> import gaitdynamics as gd

Then, use the built-in ``help`` function to view the docstring for any
function or module:

  >>> help(gd.gait.step_symmetry)

Modules
-------
gait
    Calculate various types of gait dynamics (cadence, symmetry etc.)
peak
    Detect peaks and valleys in a signal
plot
    Wrapper functions for creating simple graphs
signal
    Signal processing tools such as filtering and cross-correlation
utils
    General utility functions used throughout the package
"""

import gaitdynamics.gait
import gaitdynamics.peak
import gaitdynamics.plot
import gaitdynamics.signal
import gaitdynamics.utils
