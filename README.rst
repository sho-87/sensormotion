`Installation`_ | `Requirements`_ | `Usage`_ | `Contribution`_ | `Getting Help`_

Sensor Motion
=============

.. image:: https://img.shields.io/pypi/pyversions/sensormotion.svg
   :alt: PyPI - Python Version
   :target: https://pypi.org/project/sensormotion/

.. image:: https://img.shields.io/pypi/v/sensormotion.svg
   :alt: PyPI
   :target: https://pypi.org/project/sensormotion/

.. image:: https://img.shields.io/github/issues/sho-87/sensormotion.svg
   :alt: GitHub issues
   :target: https://github.com/sho-87/sensormotion/issues

.. image:: https://readthedocs.org/projects/sensormotion/badge/?version=latest
   :target: http://sensormotion.readthedocs.io

.. image:: https://badges.gitter.im/gitterHQ/gitter.png
   :target: https://gitter.im/sensormotion/lobby

.. inclusion-marker-main-readme

Python package for analyzing sensor-collected human motion data (e.g.
physical activity levels, gait dynamics).

Dedicated accelerometer devices, such as those made by Actigraph,
usually bundle software for the analysis of the sensor data. In my work
I often collect sensor data from smartphones and have not been able to
find any comparable analysis software.

This Python package allows the user to extract human motion data, such
as gait/walking dynamics, directly from accelerometer signals.
Additionally, the package allows for the calculation of physical
activity (PA) or moderate-to-vigorous physical activity (MVPA) counts,
similar to activity count data offered by companies like Actigraph.

Installation
------------

You can install this package using ``pip``:

::

    pip install sensormotion

Requirements
------------

This package has the following dependencies, most of which are just
Python packages:

- **Python 3.x**

  - The easiest way to install Python is using the
    `Anaconda <https://www.continuum.io/downloads>`__ distribution, as it
    also includes the other dependencies listed below

  - Python 2.x has not been tested, so backwards compatibility is not
    guaranteed

- **numpy**

  - Included with Anaconda. Otherwise, install using pip (``pip install numpy``)

- **scipy**

  - Included with Anaconda. Otherwise, install using pip (``pip install scipy``)

- **matplotlib**

  - Included with Anaconda. Otherwise, install using pip (``pip install matplotlib``)

Usage
-----

Here is brief example of extracting step-based metrics from raw
vertical acceleration data:

Import the package:

.. code:: python

    import sensormotion as sm

If you have a vertical acceleration signal ``x``, and its corresponding
time signal ``t``, we can begin by filtering the signal using a low-pass
filter:

.. code:: python

    b, a = sm.signal.build_filter(frequency=10,
                                  sample_rate=100,
                                  filter_type='low',
                                  filter_order=4)

    x_filtered = sm.signal.filter_signal(b, a, signal=x)

.. image:: images/filter.png

Next, we can detect the peaks (or valleys) in the filtered signal, which
gives us the time and value of each detection. Optionally, we can
include a plot of the signal and detected peaks/valleys:

.. code:: python

    peak_times, peak_values = sm.peak.find_peaks(time=t, signal=x_filtered,
                                                 peak_type='valley',
                                                 min_val=0.6, min_dist=30,
                                                 plot=True)

.. image:: images/peak_detection.png

From the detected peaks, we can then calculate step metrics like cadence
and step time:

.. code:: python

    cadence = sm.gait.cadence(time=t, peak_times=peak_times, time_units='ms')
    step_mean, step_sd, step_cov = sm.gait.step_time(peak_times=peak_times)

Physical activity counts and intensities can also be calculated from the acceleration data:

.. code:: python

    x_counts = sm.pa.convert_counts(x, time, integrate='simpson')
    y_counts = sm.pa.convert_counts(y, time, integrate='simpson')
    z_counts = sm.pa.convert_counts(z, time, integrate='simpson')
    vm = sm.signal.vector_magnitude(x_counts, y_counts, z_counts)
    categories, time_spent = sm.pa.cut_points(vm, set_name='butte_preschoolers', n_axis=3)

.. image:: images/pa_counts.png 

For a more in-depth tutorial, and more workflow examples, please take a look
at the `tutorial <https://github.com/sho-87/sensormotion/blob/master/tutorial.ipynb>`__.

I would also recommend looking over the
`documentation <http://sensormotion.readthedocs.io>`__ to see other
functionalities of the package.

Contribution
------------

I work on this package in my spare time, on an "as needed" basis for my
research projects. However, pull requests for bug fixes and new features
are always welcome!

Please see the `develop
branch <https://github.com/sho-87/sensormotion/tree/develop>`__ for the
development version of the package, and check out the `issues
page <https://github.com/sho-87/sensormotion/issues>`__ for bug reports
and feature requests.

Getting Help
------------

You can find the full documentation for the package
`here <http://sensormotion.readthedocs.io>`__

Python's built-in help function will show documentation for any module
or function: ``help(sm.gait.step_time)``

You're encouraged to post questions, bug reports, or feature requests as
an `issue <https://github.com/sho-87/sensormotion/issues>`__

Alternatively, ask questions on
`Gitter <https://gitter.im/sensormotion/lobby>`__
