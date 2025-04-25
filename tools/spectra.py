# Spectral Analysis functions
## Luke Colosi | lcolosi@ucsd.edu 

#--- Spectral Uncertainities ---# 
def spectral_uncer(N, alpha, psd, estimator, M = []):

    """
    Computes the upper and lower bounds of the 95% confidence interval 
    for power density spectrum using either the Welch method with overlapping 
    hanning windowed segments or the autocovariance function approach (also
    using a hanning window).   
    
    Parameters
    ----------
    N : int
        Length of segment (total number of data points per segment or record for computing the full non-truncated 
        autocovariance function).  
    alpha : float
        Significance level ranging between 0 to 1. For 95% confidence interval, set alpha = 0.05.
    psd : numpy array
        Normalized Power Spectral Density function.
    estimator: str
        Specifies whether the spectral estimate is obtained by the fourier transform of the physical data record 
        or the autocovariance function. This will impact the degrees of freedom in your uncertainty estimate.
        Options include: 'data' or 'autocov'
    M : int
        The half-width for autovariance function segmenting procedure. In th
    
    Returns
    -------
    error : tuple
        Upper and lower bounds of % confidence interval.
    CI : numpy array
        Matrix with upper and lower bounds of confidence interval as a function of frequency.
    error_ratio : float
        Error bar ratio between upper and lower CI.
    """
    
    # Import libraries
    import numpy as np
    from scipy.stats import chi2

    # Compute Degrees of Freedom (assuming hanning window)
    if estimator == 'data':
        nu = (36 / 19) * (2 * N - 1) 
    elif estimator == 'autocov':
        nu = (8/3) * (N/M)
    
    # Compute the Upper and Lower bounds of % confidence interval
    error_high = nu / chi2.ppf(alpha / 2, nu)
    error_low = nu / chi2.ppf(1 - alpha / 2, nu)
    
    # Save upper and lower bounds
    error = (error_low, error_high)
    
    # Compute error bar as a function of frequency (scale properly for the y-axis location of the PSD function)
    CI = np.column_stack((error_low * psd, error_high * psd))
    
    # Compute error bar ratio
    error_ratio = error_high / error_low
    
    return error, CI, error_ratio







#--- 1D Power Spectrum from the Autocovariance function ---# 
def spectra_autocov(autocov_pos, autocov_neg, M, dt, units): 
    
    """
    spectra_autocov(autocov_pos, autocov_neg, M, dt)
        
        Function to compute the power spectrum from the unbiased estimate of the autocovariance function. 
        A hanning window is applied to the segmented autocovariance function to downweight the points 
        at the tails of the autocovariance function which has less averaging. By "segmented", a band centered
        on the zero lag is used for computing the power spectrum to exclude larger lags with larger uncertainties. 
        
        Parameters
        ----------
        autocov_pos : numpy array 
            Unbiased autocovariance function for positive lag times. 
        autocov_neg : numpy array
            Unbiased autocovariance function for negative lag times. 
        N : int
            Number of data points used the compute the autocovariance function. 
        M : int
            Nominaly the half width of window length of autocovariance that will be used to segment the central peak 
            of Autocovariance for computing spectra. Make sure to account for the lag time step of the 
            autocovariance function when choosing M.  
        dt : float
            Specifies the time or spatial interval between measurements. 
        units : str
            Specifies the units of the frequency vector. Options: 'Hz' (cyclical frequency) or 'rad/s' (radian frequency).
    
        Returns
        -------
        fft_autocov : array
            Fourier Transform of Autocovariance 
        amp_norm : array 
            Normalized Power Spectral Density Function 
        freq : array 
            Frequencies corresponding to Power Density Spectrum 
        
        Libraries necessary to run function
        -----------------------------------
        import numpy as np
    
    """
    
    # Import libraries
    import numpy as np
    from spectra import spectral_uncer

    ###########################################################################
    ## STEP #1 - Segment the autocovariance function 
    ###########################################################################

    # Set the number of data points used in the autocovariance estimate 
    N = np.count_nonzero(~np.isnan(autocov_pos))

    # Select the band of autocovariance from specified M value (Nominally the half-width)
    autocov_pos_seg = autocov_pos[:M+1]
    autocov_neg_seg = autocov_neg[-M:]
    
    # Combine the positive and negative autocovariance bands
    autocov_func_seg = np.append(autocov_neg_seg, autocov_pos_seg)
    
    # Compute segment length 
    p = len(autocov_func_seg)

    ###########################################################################
    ## STEP #2 - Set fundamental parameters for computing spectrum 
    ###########################################################################

    # Compute the frequency resolution
    if units == 'Hz':
        df = 1 / (p * dt)
    elif units == 'rad/s':
        df = (2 * np.pi) / (p * dt)
    else:
        raise ValueError("Invalid unit. Use 'Hz' or 'rad/s'.")
    
    # Compute number of positive frequencies
    if p % 2 == 0:
        L = p // 2 + 1 
    else:
        L = (p + 1) // 2

    # Compute the period or wavelength of the fundamental frequency
    T = p * dt

    # Compute frequency vector
    if p % 2 == 0:
        if units == 'Hz': 
            f = np.arange(p // 2 + 1) / T
        else: 
            f = (2 * np.pi * np.arange(p // 2 + 1)) / T
    else:
        if units == 'Hz':
            f = np.arange((p + 1) // 2) / T  
        else:
            f= (2 * np.pi * np.arange((p + 1) // 2)) / T

    ###########################################################################
    ## STEP #3 - Compute 1D frequency spectrum and its uncertainty
    ###########################################################################
    
    # Apply a normalized window to the segmented autocovariance function
    window = np.hanning(p)*np.sqrt(p/np.sum(np.hanning(p)**2))
    autocov_window = autocov_func_seg*window
    
    # Compute the fourier transform of the windowed autocovariance function  
    fft_autocov = np.fft.fft(autocov_window)

    # Compute the squared amplitudes (recall that the fourier transform of the autocovariance function are the squared amplitudes)
    amp = abs(fft_autocov)

    # Grab positive frequencies for single-sided PSD
    amp_pos = amp[:L]
    
    # Double the amplitude for positive frequencies to conserve variance
    if p % 2 == 0:
        amp_pos[1:-1] *= 2
    else:
        amp_pos[1:] *= 2
        
    # Normalize power spectral density
    psd = amp_pos/(p**2 * df)

    # Compute the variance in frequency space
    variance = np.sum(psd * df)

    # Compute 95% confidence interval
    _, CI, _ = spectral_uncer(N, 0.05, psd, 'autocov', M)
    
    return fft_autocov, psd, f, CI, variance





#--- 1D Power Spectrum without the Welch Method ---# 
def spectrum1D_frequency_nonwelch(data, dt, units):
    """
    Computes the 1D power density spectrum without the Welch method.
    
    Parameters
    ----------
    data : numpy array
        Time or spatial data series. Data must be evenly spaced.
    dt : float
        Time or spatial interval between measurements.
    units : str
        Specifies the units of the frequency vector. Options: 'Hz' (cyclical frequency) or 'rad/s' (radian frequency).
    
    Returns
    -------
    psd : numpy array
        Normalized power spectral density function.
    f : numpy array
        Frequency in specified units.
    """

    import numpy as np
    from scipy.signal import detrend

    # Number of data points
    N = len(data)
    
    # Frequency resolution
    if units == 'Hz':
        df = 1 / (N * dt)
    elif units == 'rad/s':
        df = (2 * np.pi) / (N * dt)
    else:
        raise ValueError("Invalid unit. Use 'Hz' or 'rad/s'.")
    
    # Compute number of positive frequencies
    L = N // 2 + 1 if N % 2 == 0 else (N + 1) // 2
    
    # Compute the period of the fundamental frequency
    T = N * dt
    
    # Compute frequency vector
    if N % 2 == 0:
        f = np.arange(N // 2 + 1) / T if units == 'Hz' else (2 * np.pi * np.arange(N // 2 + 1)) / T
    else:
        f = np.arange((N + 1) // 2) / T if units == 'Hz' else (2 * np.pi * np.arange((N + 1) // 2)) / T
    
    # Detrend time series
    data_dt = detrend(data)
    
    # Compute FFT of the time series
    fft_data = np.fft.fft(data_dt)
    
    # Take squared modulus of the Fourier coefficients
    amp = np.abs(fft_data) ** 2
    
    # Grab positive frequencies for single-sided PSD
    amp_pos = amp[:L]
    
    # Double the amplitude for positive frequencies to conserve variance
    if N % 2 == 0:
        amp_pos[1:-1] *= 2
    else:
        amp_pos[1:] *= 2
    
    # Normalize power spectral density
    psd = amp_pos / (N**2 * df)
    
    return psd, f


#--- 1D Power Spectrum with the Welch Method ---# 
def spectrum1D_frequency(data, dt, M, units):

    """
    Function for computing the 1D power density spectrum with the Welch method.
    This function is written notationally for time series, but can be applied to spatial data.
    The 1D frequency spectrum is computed by Hanning windowing segments of the data array with 50% overlap.
    
    Parameters
    ----------
    data : Time or spatial data series. Data must be evenly spaced (NaNs must be interpolated).
    dt : Time or spatial interval between measurements.
    M : Number of windows.
    units : 'Hz' (cyclical frequency) or 'rad/s' (radian frequency).
    
    Returns
    -------
    psd : Normalized power spectral density function.
    f : Frequency in units specified by units variable.
    CI : 95% confidence interval.
    variance : Dictionary containing the variance in the time and frequency domains.
    """

    # Import libraries
    import numpy as np
    from spectra import spectral_uncer
    from scipy.signal import hann, detrend

    ###########################################################################
    ## STEP #1 - Set fundamental parameters for computing spectrum
    ###########################################################################

    N = len(data)                 # Number of data points of entire time series
    p = N // M                    # Number of data points within a window

    # Compute frequency resolution
    if units == 'Hz':
        df = 1 / (p * dt)
    elif units == 'rad/s':
        df = 2 * np.pi / (p * dt)

    # Compute number of positive frequencies
    if p % 2 == 0:
        L = p // 2 + 1
    else:
        L = (p - 1) // 2

    # Compute the period of the fundamental frequency (lowest frequency)
    T = p * dt

    # Compute frequency vector (units: Hz or rad/s)
    if p % 2 == 0:
        if units == 'Hz':
            f = (1 / T) * np.arange(0, p // 2 + 1)
        elif units == 'rad/s':
            f = (2 * np.pi / T) * np.arange(0, p // 2 + 1)
    else:
        if units == 'Hz':
            f = (1 / T) * np.arange(0, (p - 1) // 2)
        elif units == 'rad/s':
            f = (2 * np.pi / T) * np.arange(0, (p - 1) // 2)

    ###########################################################################
    ## STEP #2 - Segment data with 50% overlap
    ###########################################################################

    nseg = M + M - 1              # Compute number of segments including 50% overlap

    # Initialize array for splitting time series into windows with 50% overlap
    data_seg_n = data[:M*p].reshape((p, M), order='F')  # Segment original data set

    data_seg_50 = []
    for iseg in range(M - 1):
        ind_i = int(p * iseg + (p / 2))
        ind_f = int(ind_i + p)
        if ind_f <= len(data):
            data_seg_50.append(data[ind_i:ind_f])

    if data_seg_50:
        data_seg_50 = np.stack(data_seg_50, axis=1)
        data_seg_n = np.concatenate((data_seg_n, data_seg_50), axis=1)

    ###########################################################################
    ## STEP #3 - Remove linear trend for each segment and apply hanning window
    ###########################################################################

    # Obtain a hanning window:
    window = hann(p) * np.sqrt(p / np.sum(hann(p)**2))

    # Preallocate windowed detrended segmented data array
    data_seg_w = np.zeros_like(data_seg_n)

    for iseg in range(data_seg_n.shape[1]):
        data_seg_w[:, iseg] = detrend(data_seg_n[:, iseg]) * window

    ###########################################################################
    ## STEP #4 - Compute mean 1D frequency spectrum
    ###########################################################################

    spec_sum = np.zeros(p)                 # Preallocate spectrum summation array
    cn = np.zeros(p)                       # Preallocate counter
    variance = {'time': np.zeros(nseg)}   # Preallocate variance in time domain

    for iseg in range(nseg):
        fft_data_seg = np.fft.fft(data_seg_w[:, iseg])          # Fourier transform data
        amp = np.abs(fft_data_seg)**2                           # Compute amplitudes
        amp_norm = amp / (p**2) / df                            # Normalize amplitudes

        variance['time'][iseg] = np.var(data_seg_w[:, iseg])    # Variance in time domain

        spec_sum += amp_norm                                    # Sum spectrum
        cn += 1                                                 # Update counter

    m_spec = spec_sum / cn                                      # Compute mean spectrum
    psd = m_spec[:L]                                            # Grab positive frequencies

    # Double the amplitude for positive frequencies to conserve variance
    if N % 2 == 0:
        psd[1:-1] *= 2
    else:
        psd[1:] *= 2

    # Compute the variance in frequency space
    variance['freq'] = np.sum(psd * df)

    # Compute 95% confidence interval
    CI = spectral_uncer(M, 0.05, psd)

    return psd, f, CI, variance



