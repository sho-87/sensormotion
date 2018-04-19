"""
sensormotion
============

Provides tools for analyzing sensor-collected human motion data. This
includes, for example, estimation of gait dynamics from accelerometer data,
and conversion to physical activity (MVPA) counts from acceleration. Also
contains a few useful functions for pre-processing and visualizing
accelerometer signals.

This package was primarily developed for use on Android sensor data collected
at the Attentional Neuroscience Lab (University of British Columbia).

Documentation
-------------
Documentation is available via docstrings provided with the code, and an
online API reference found at
`ReadTheDocs <http://sensormotion.readthedocs.io>`_.

To view documentation for a function or module, first make sure the package
has been imported:

  >>> import sensormotion as sm

Then, use the built-in ``help`` function to view the docstring for any
function or module:

  >>> help(sm.gait.step_symmetry)

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

import sensormotion.gait
import sensormotion.peak
import sensormotion.plot
import sensormotion.signal
import sensormotion.utils
