"""
Calculate physical activity (PA) levels with conversion to activity counts.

Functions for converting raw sensor data to physical activity (PA) or
moderate-to-vigorous physical activity (MVPA) counts, similar to those given
by dedicated accelerometers such as Actigraph devices.
"""

import matplotlib.pyplot as plt
import numpy as np
import sensormotion.signal

from scipy.integrate import simps, trapz


def convert_counts(x, time, time_scale='ms', epoch=60, rectify='full',
                   integrate='simpson', plot=False, fig_size=(10, 5)):
    """
    Convert acceleration to physical activity (PA) counts.

    Given an acceleration signal over a **single** axis, integrate the signal
    for each time window (epoch). The area under the curve for each epoch is
    the physical activity count for that time period.

    Parameters
    ----------
    x : ndarray
        Acceleration signal to be converted to PA counts.
    time : ndarray
        Time signal associated with `x`.
    time_scale : {'ms', 's'}, optional
        The unit that `time` is measured in. Either seconds (s) or
        milliseconds (ms).
    epoch : int, optional
        The duration of each time window in seconds. Counts will be calculated
        over this period. PA counts are usually measured over 60 second epochs.
    rectify : {'full', 'half'}, optional
        Type of rectifier to use on the input acceleration signal. This is to
        ensure that PA counts take into consideration negative acceleration
        values. Full-wave rectification turns all negative values into
        positive ones. Half-wave rectification sets all negative values to
        zero.
    integrate : {'simpson', 'trapezoid'}, optional
        Integration method to use for each epoch.
    plot : bool, optional
        Toggle to display a plot of PA counts over time.
    fig_size : tuple, optional
        If plotting, set the width and height of the resulting figure.

    Returns
    -------
    counts : ndarray
        PA count values for each epoch.
    """

    assert len(x) == len(time), 'signal and time must be the same length'
    assert np.all(np.diff(time) > 0), 'time signal is not fully ascending'
    assert integrate == 'simpson' or integrate == 'trapezoid', \
        'integrate method must either be simpson or trapezoid'

    x = np.asarray(x)
    time = np.asarray(time)

    # convert time to seconds
    if time_scale == 'ms':
        time = time/1000
    elif time_scale == 's':
        time = time

    # calculate time diff
    time = time - time[0]

    assert max(time) > epoch, 'length of signal time shorter than epoch size'

    # rectify signal
    x = sensormotion.signal.rectify_signal(x, rectify)

    # interpolate missing times values to get exact epochs
    boundary_count = int(max(time) / epoch) + 1
    boundary_times = [i*epoch for i in range(boundary_count)]
    missing_times = np.setdiff1d(boundary_times, time)  # epoch times to interp

    x = np.append(x, np.interp(missing_times, time, x))  # interpolate x values
    time = np.append(time, missing_times)

    # sort new time and signal arrays together
    sort_idx = time.argsort()
    time = time[sort_idx]
    x = x[sort_idx]

    # get index of each epoch/boundary value for slicing
    boundary_idx = np.where(np.isin(time, boundary_times))[0]

    # integrate each epoch using Simpson's rule
    counts = np.ones(len(boundary_idx) - 1)  # preallocate array

    for i in range(len(counts)):
        lower = boundary_idx[i]
        upper = boundary_idx[i+1] + 1  # upper bound should be inclusive

        cur_signal = x[lower:upper]
        cur_time = time[lower:upper]

        if integrate == 'simpson':
            counts[i] = simps(cur_signal, cur_time)
        elif integrate == 'trapezoid':
            counts[i] = trapz(cur_signal, cur_time)

    # plot counts
    if plot:
        f, ax = plt.subplots(1, 1, figsize=fig_size)

        ax.bar(boundary_times[1:], counts, width=epoch-1)

        plt.xticks(boundary_times[1:],
                   ['{} - {}'.format(boundary_times[i], boundary_times[i+1])
                    for i, x in enumerate(boundary_times[1:])])

        for tick in ax.get_xticklabels():
            tick.set_rotation(45)

        plt.suptitle('Physical activity counts', size=16)
        plt.xlabel('Time window (seconds)')
        plt.ylabel('PA count')
        plt.show()

    return counts
