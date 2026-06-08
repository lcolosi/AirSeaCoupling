# General filtering functions
## Luke Colosi | lcolosi@ucsd.edu 

#--- Lanczos low pass filter ---# 
def lanczos_lowpass_filter(x, cutoff, window, dt=1.0, axis=0):
    """
    Apply a Lanczos low-pass filter to time series data.

    Parameters
    ----------
    x : ndarray
        Input data (time along `axis`).
    cutoff : float
        Cutoff frequency in cycles per unit time (e.g. 1/90 for 90-day cutoff if dt=1 day).
    window : int
        Half-width of the Lanczos window (number of points on either side of center).
        The total kernel length = 2*window + 1.
    dt : float, optional
        Sampling interval (default = 1.0).
    axis : int, optional
        Axis along which to apply the filter (default = 0).

    Returns
    -------
    y : ndarray
        Low-pass filtered time series (same shape as input).
    """

    # Import libraries
    import numpy as np
    from scipy.signal import convolve

    # Nyquist frequency = maximum resolvable frequency for given sampling rate
    nyquist = 0.5 / dt

    # Normalized cutoff frequency (0–1, relative to Nyquist)
    fc = cutoff / nyquist

    # Create symmetric time index for kernel, e.g. [-window, ..., 0, ..., window]
    n = np.arange(-window, window + 1)

    # Ideal sinc low-pass kernel (infinite length, truncated here)
    h = np.sinc(2 * fc * n)

    # Lanczos window (extra sinc taper to suppress sidelobes)
    lanczos = np.sinc(n / window)

    # Multiply ideal kernel by Lanczos window → final FIR kernel
    kernel = h * lanczos

    # Normalize so kernel sums to 1 (ensures mean is preserved)
    kernel /= kernel.sum()

    # Apply convolution along the chosen axis
    # If axis=0 (time dimension is rows), expand kernel to column vector
    y = convolve(x, kernel[:, None] if axis == 0 else kernel, mode="same")

    return y


#--- Low-pass Butterworth Digital Filter ---# 
def butter_lowpass_filter(x, cutoff, dt=1.0, order=4, axis=0):

    """
    Apply a zero-phase Butterworth low-pass filter to time series data.

    Parameters
    ----------
    x : ndarray
        Input data (time along `axis`).
    cutoff : float
        Cutoff frequency in cycles per unit time (e.g. 1/90 for 90-day cutoff if dt=1 day).
    dt : float, optional
        Sampling interval (default = 1.0).
    order : int, optional
        Filter order (default = 4).
        Higher order → sharper cutoff, but more computation and possible ringing.
    axis : int, optional
        Axis along which to filter (default = 0).

    Returns
    -------
    y : ndarray
        Low-pass filtered time series (same shape as input).
    """

    # Import libraries
    from scipy.signal import butter, filtfilt

    # Nyquist frequency = maximum resolvable frequency
    nyquist = 0.5 / dt

    # Normalized cutoff frequency (0–1, relative to Nyquist)
    Wn = cutoff / nyquist

    # Design Butterworth filter (returns numerator b and denominator a)
    b, a = butter(order, Wn, btype="low")

    # Apply filter forward and backward with filtfilt
    # → ensures zero phase shift (important in geophysical time series)
    y = filtfilt(b, a, x, axis=axis)

    return y


