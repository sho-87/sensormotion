"""
Calculate gait dynamics.

Functions for the calculation of variance gait dynamics from acceleration
data (e.g. step symmetry, cadence).
"""

from __future__ import print_function, division

import numpy as np


def cadence(time, peak_times, time_units="ms"):
    """
    Calculate cadence of the current signal.

    Cadence (steps per minute) can be estimated by detecting peaks in the
    acceleration vector. Given 1) the duration of the signal and 2) the number
    of steps/peaks in the signal, we can calculate an estimate of steps per
    minute.

    Peak detection provides number of steps within the time frame of the
    signal. This is then extrapolated from milliseconds to minutes to estimate
    cadence.

    Parameters
    ----------
    time : ndarray
        Time vector of the original acceleration signal. Used to calculate
        duration of the input signal.
    peak_times : ndarray
        Time of each peak, returned by :func:`sensormotion.peak.find_peaks`.
        This provides the number of steps within the timeframe of the signal.
    time_units : {'ms', 's'}, optional
        Units of the time signal.

    Returns
    -------
    cadence : float
        Estimated cadence for the input signal.
    """

    n = step_count(peak_times)

    # Convert duration to seconds
    if time_units == "ms":
        duration = (time.max() - time.min()) / 1000
    elif time_units == "s":
        duration = time.max() - time.min()

    steps_per_min = (n / duration) * 60

    return steps_per_min


def step_count(peak_times):
    """
    Count total number of steps in the signal.

    This is simply the number of peaks detected in the signal.

    Parameters
    ----------
    peak_times : ndarray
        Times of the peaks detected by :func:`sensormotion.peak.find_peaks`.

    Returns
    -------
    step_count : int
        Number of steps/peaks in the signal.
    """

    return len(peak_times)


def step_regularity(autocorr_peak_values):
    """
    Calculate step and stride regularity from autocorrelation peak values.

    Step and stride regularity measures based on
    `Moe-Nilssen (2004) - Estimation of gait cycle characteristics by trunk
    accelerometry
    <http://www.jbiomech.com/article/S0021-9290(03)00233-1/abstract>`_.

    If calculating regularity from acceleration in the vertical axis, this
    function receives the detected peaks from the vertical axis
    autocorrelation.

    However, if calculating regularity from lateral axis
    acceleration, you should pass in *both* peaks and valleys from the
    autocorrelation of the lateral axis.

    **Step regularity:**

    Perfect step regularity will be 1.0 for vertical axis autocorrelation
    (the larger the better, capped at 1.0).

    For the lateral axis, perfect regularity is -1.0 (the smaller the
    better, capped at -1.0).

    **Stride regularity:**

    Perfect stride regularity will be 1.0 for vertical axis autocorrelation
    (the larger the better, capped at 1.0).

    Lateral axis sign and interpretation are the same as the vertical axis.

    Parameters
    ----------
    autocorr_peak_values : ndarray
        Values of the autocorrelation peaks/valleys detected by
        :func:`sensormotion.peak.find_peaks`. This should contain only peak
        values when looking at the vertical axis, and both peak and valley
        values when looking at the lateral axis.

    Returns
    -------
    step_reg : float
        Step regularity. Value is capped at 1.0 or -1.0 depending on the
        axis of interest.
    stride_reg : float
        Stride regularity. Capped at 1.0 for both vertical and lateral axes.
    """

    peaks_half = autocorr_peak_values[autocorr_peak_values.size // 2 :]

    assert len(peaks_half) >= 3, (
        "Not enough autocorrelation peaks detected. Plot the "
        "autocorrelation signal to visually inspect peaks"
    )

    ac_lag0 = peaks_half[0]  # autocorrelation value at lag 0
    ac_d1 = peaks_half[1]  # first dominant period i.e. a step (left-right)
    ac_d2 = peaks_half[2]  # second dominant period i.e. a stride (left-left)

    step_reg = ac_d1 / ac_lag0
    stride_reg = ac_d2 / ac_lag0

    return step_reg, stride_reg


def step_symmetry(autocorr_peak_values):
    """
    Calculate step symmetry from autocorrelation peak values.

    Step symmetry measures based on `Moe-Nilssen (2004) - Estimation of gait
    cycle characteristics by trunk accelerometry
    <http://www.jbiomech.com/article/S0021-9290(03)00233-1/abstract>`_.

    If calculating symmetry from acceleration in the vertical axis, this
    function receives the detected peaks from the vertical axis
    autocorrelation.

    However, if calculating symmetry from lateral axis
    acceleration, you should pass in *both* peaks and valleys from the
    autocorrelation of the lateral axis.

    Perfect step symmetry is 1.0 for the vertical axis - larger values are
    more symmetric, capped at 1.0.

    Perfect step symmetry is -1.0 for the lateral axis - smaller values are
    more symmetric, capped at -1.0.

    Parameters
    ----------
    autocorr_peak_values : ndarray
        Values of the autocorrelation peaks/valleys detected by
        :func:`sensormotion.peak.find_peaks`. This should contain only peak
        values when looking at the vertical axis, and both peak and valley
        values when looking at the lateral axis.

    Returns
    -------
    step_sym : float
        Step symmetry. Value is capped at 1.0 or -1.0 depending on the
        axis of interest.
    """

    peaks_half = autocorr_peak_values[autocorr_peak_values.size // 2 :]

    assert len(peaks_half) >= 3, (
        "Not enough autocorrelation peaks detected. Plot the "
        "autocorrelation signal to visually inspect peaks"
    )

    ac_d1 = peaks_half[1]  # first dominant period i.e. a step (left-right)
    ac_d2 = peaks_half[2]  # second dominant period i.e. a stride (left-left)

    # Always divide smaller peak by the larger peak
    if abs(ac_d1) > abs(ac_d2):
        step_sym = ac_d2 / ac_d1  # Preserve sign by not using abs()
    else:
        step_sym = ac_d1 / ac_d2  # Preserve sign by not using abs()

    return step_sym


def step_time(peak_times):
    """
    Calculate step timing information.

    Step timing can be calculated from the peak times of the original
    acceleration signal. This includes mean time between steps,
    standard deviation of step time, and the coefficient of
    variation (sd/mean).

    Parameters
    ----------
    peak_times : ndarray
        Times of the peaks detected by :func:`sensormotion.peak.find_peaks`.

    Returns
    -------
    step_time_mean : float
        Mean time between all steps/peaks in the signal.
    step_time_sd : float
        Standard deviation of the distribution of step times in the signal.
    step_time_cov : float
        Coefficient of variation. Calculated as sd/mean.
    """

    peak_time_differences = np.diff(peak_times)
    peak_time_mean = np.mean(peak_time_differences)
    peak_time_sd = np.std(peak_time_differences)
    peak_time_cov = peak_time_sd / peak_time_mean

    return peak_time_mean, peak_time_sd, peak_time_cov
