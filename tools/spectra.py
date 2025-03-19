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

