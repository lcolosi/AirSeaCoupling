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


def spectral_slope(f, E, fmin, fmax):
    """
    Function for computing the spectral slope in log space of a power spectral
    density function within a specified frequency subrange.

    Parameters
    ----------
    f : ndarray
        Cyclical frequencies corresponding to the PSD.
    E : ndarray
        Normalized power spectral density function.
    fmin : float
        Lower frequency limit for subrange.
    fmax : float
        Upper frequency limit for subrange.

    Returns
    -------
    m : float
        Spectral slope (slope of log-log linear fit).
    mm : float
        Uncertainty of the spectral slope.
    yfit : ndarray
        Fitted PSD values in linear space.
    f_range : ndarray
        Frequency subrange in linear space.
    """

    # Import libraries
    import numpy as np

    # Select frequency subrange
    idx = (f >= fmin) & (f <= fmax) & np.isfinite(f) & np.isfinite(E)
    fi = np.log10(f[idx])
    Ei = np.log10(E[idx])
    
    # Preform least squares fit (linear regression in log-log space)
    A = np.vstack([np.ones_like(fi), fi]).T
    coef, residuals, _, _ = np.linalg.lstsq(A, Ei, rcond=None)

    # Set slope from solution of LSF
    m = coef[1]
    
    # Compute unweighted least square fit model
    yfit_log = A @ coef

    # Map frequency subrange and yfit back to linear space
    yfit = 10**yfit_log
    f_range = 10**fi

    # Initialize variables for uncertainty estimation
    M = len(Ei)
    parameters = 2

    # Compute uncertainty of data
    if M <= parameters:
        mm = np.nan
    else:

        # Compute variance uncertainty of data in 
        var_fit = np.sum((Ei - yfit_log)**2) / (M - parameters)
        std_fit = np.sqrt(var_fit)

        # Compute standard error of the spectral slope
        delta = M * np.sum(f_range**2) - (np.sum(f_range))**2
        mm = std_fit * np.sqrt(M / delta)

    return m, mm, yfit, f_range



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
    from scipy.signal.windows import hann
    from scipy.signal import detrend

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

    # Compute number of segments including 50% overlap
    nseg = M + M - 1     

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
    _, CI, _ = spectral_uncer(M, 0.05, psd, 'data')

    return psd, f, CI, variance

#--- convert a 2D spectrum from Cartesian (kx, ky) to polar (r, θ) coordinates ---# 
def xy2rt(Im, kx, ky):

    """
    Function that takes a 2D wavenumber spectrum in kx-ky coordinates and
    returns it in polar coordinates r-theta.

    Parameters
    ----------
    Im : 2D array
        2D wavenumber spectrum in Cartesian coordinates (kx, ky).
    kx : 1D array
        x-component of the wavenumber vector.
    ky : 1D array
        y-component of the wavenumber vector.

    Returns
    -------
    Z : 2D array
        Wavenumber spectrum interpolated (nearest neighbor) onto the xZ-yZ Cartesian grid.
    t : 1D array
        Theta coordinate in polar grid. Units: radians.
    r : 1D array
        Radial coordinate in polar grid. Units: same as kx/ky.
    xZ : 2D array
        x-coordinate in Cartesian grid mapped from polar space.
    yZ : 2D array
        y-coordinate in Cartesian grid mapped from polar space.
    """

    # Import functions 
    import numpy as np
    from scipy.interpolate import interp2d

    # Set wavenumber resolution
    dkx = kx[1] - kx[0]
    dky = ky[1] - ky[0]

    # Determine minimum and maximum wavenumber for setting r grid
    min_k = min(dkx, dky)
    max_k = max(np.max(np.abs(kx)), np.max(np.abs(ky)))

    # Set theta grid (from -π to π)
    t = np.linspace(-np.pi, np.pi, 361)

    # Set radial grid from 0 to max_k with resolution min_k / 2
    r = np.arange(min_k, max_k + min_k / 2, min_k / 2)

    # Generate polar meshgrid
    th, rh = np.meshgrid(t, r)

    # Convert polar grid to Cartesian coordinates
    xZ, yZ = rh * np.cos(th), rh * np.sin(th)

    # Interpolate original spectrum to new Cartesian grid
    # We use 'nearest' interpolation as in original MATLAB code
    interp_func = interp2d(kx, ky, Im, kind='nearest')
    Z = interp_func(xZ, yZ)

    return Z, t, r, xZ, yZ


#--- Compute the 2D Wavenumber Spectrum with Welch method ---# 
def spectrum2D_wavenumber(data, x, y, dx, dy, Mx, My):
    """
    Function for computing the 2D wavenumber spectrum from 2-dimensional
    spatial data. The 2D wavenumber spectrum is computed by hanning
    windowing 2D segments of the data array with 50% overlap.

    Parameters
    ----------
    data : 2D np.ndarray
        Evenly spaced 2D spatial data.
    x : np.ndarray
        x-coordinate vector.
    y : np.ndarray
        y-coordinate vector.
    dx : float
        Distance between adjacent x-coordinate measurements.
    dy : float
        Distance between adjacent y-coordinate measurements.
    Mx : int
        Number of segments to split the x-axis into.
    My : int
        Number of segments to split the y-axis into.

    Returns
    -------
    m_spec : 2D np.ndarray
        Averaged power spectrum in Cartesian coordinates (units: m^2/(rad/m)^2).
    kx : np.ndarray
        x-component of the wavenumber vector (units: rad/m).
    ky : np.ndarray
        y-component of the wavenumber vector (units: rad/m).
    m_spec_pol : 2D np.ndarray
        Averaged power spectrum in polar coordinates (units: m^2/(rad/m * rad)).
    k : np.ndarray
        Wavenumber magnitude vector (units: rad/m).
    theta : np.ndarray
        Wave direction vector (units: radians).
    spec_omni : np.ndarray
        Omni-directional spectrum (units: m^2/(rad/m)).
    """

    # Import functions 
    import numpy as np
    from scipy.signal import windows
    from scipy.fft import fft2, fftshift
    from plane_lsf import plane_lsf  
    from xy2rt import xy2rt          

    #-----------------------------------------------------------------------
    # STEP #0 - Set fundamental parameters for computing spectrum
    #-----------------------------------------------------------------------

    Nx = data.shape[0]  # Number of points along x-axis
    Ny = data.shape[1]  # Number of points along y-axis
    px = Nx // Mx       # Points per segment along x-axis (integer division)
    py = Ny // My       # Points per segment along y-axis (integer division)

    # Compute wavenumber resolution along each axis (fundamental frequency)
    dkx = 2 * np.pi / (px * dx)
    dky = 2 * np.pi / (py * dy)

    # Determine FFT length for x-axis (even length preferred)
    Nxf = px if px % 2 == 0 else px - 1
    # Determine FFT length for y-axis (even length preferred)
    Nyf = py if py % 2 == 0 else py - 1

    # Compute spatial domain lengths of each segment (meters)
    Lx = Nxf * dx
    Ly = Nyf * dy

    # Construct wavenumber vectors kx and ky with proper FFT indexing and shift
    if px % 2 == 0:
        kx = (2 * np.pi / Lx) * np.concatenate((np.arange(0, Nxf // 2), np.arange(-Nxf // 2, 0)))
    else:
        kx = (2 * np.pi / Lx) * np.concatenate((np.arange(0, Nxf // 2 + 1), np.arange(-Nxf // 2, 0)))

    if py % 2 == 0:
        ky = (2 * np.pi / Ly) * np.concatenate((np.arange(0, Nyf // 2), np.arange(-Nyf // 2, 0)))
    else:
        ky = (2 * np.pi / Ly) * np.concatenate((np.arange(0, Nyf // 2 + 1), np.arange(-Nyf // 2, 0)))

    # Shift zero frequency component to center of the spectrum for visualization
    kx = fftshift(kx)
    ky = fftshift(ky)

    #-----------------------------------------------------------------------
    # STEP #1 - Segment data with 50% overlap
    #-----------------------------------------------------------------------

    # Number of segments including 50% overlapping segments
    if Mx > 1:
        nseg = Mx * My + (Mx - 1) * (My - 1)
    else:
        nseg = Mx * My + Mx * (My - 1)

    # Preallocate arrays for data segments and corresponding coordinate vectors
    data_seg = np.zeros((px, py, nseg))
    x_seg = np.zeros((px, nseg))
    y_seg = np.zeros((py, nseg))

    cn = 0  # Segment counter

    # Non-overlapping segments
    for ix in range(Mx):
        idx_xi = ix * px
        idx_xf = idx_xi + px
        for iy in range(My):
            idx_yi = iy * py
            idx_yf = idx_yi + py

            # Extract segment from data and coordinate vectors
            data_seg[:, :, cn] = data[idx_xi:idx_xf, idx_yi:idx_yf]
            x_seg[:, cn] = x[idx_xi:idx_xf]
            y_seg[:, cn] = y[idx_yi:idx_yf]

            cn += 1

    # 50% overlapping segments
    for ix in range(Mx - 1):
        idx_xi = int(ix * px + px / 2)
        idx_xf = idx_xi + px
        for iy in range(My - 1):
            idx_yi = int(iy * py + py / 2)
            idx_yf = idx_yi + py

            # Extract overlapping segment and coordinate vectors
            data_seg[:, :, cn] = data[idx_xi:idx_xf, idx_yi:idx_yf]
            x_seg[:, cn] = x[idx_xi:idx_xf]
            y_seg[:, cn] = y[idx_yi:idx_yf]

            cn += 1

    #-----------------------------------------------------------------------
    # STEP #2 - Remove linear trend from each segmented data
    #-----------------------------------------------------------------------

    data_seg_dt = np.zeros_like(data_seg)  # Preallocate detrended segments array

    for iseg in range(nseg):
        # Generate meshgrid of spatial coordinates for planar fitting
        X, Y = np.meshgrid(x_seg[:, iseg], y_seg[:, iseg], indexing='ij')

        # Perform planar least squares fit to remove linear trend
        plane_fit, _, _, _ = plane_lsf(data_seg[:, :, iseg], X, Y, method=1, weights=None)

        # Remove planar fit (linear trend) from segment
        data_seg_dt[:, :, iseg] = data_seg[:, :, iseg] - plane_fit

    #-----------------------------------------------------------------------
    # STEP #3 - Apply Hanning window to each detrended segment
    #-----------------------------------------------------------------------

    wx = windows.hann(px) * np.sqrt(px / np.sum(windows.hann(px) ** 2))
    wy = windows.hann(py) * np.sqrt(py / np.sum(windows.hann(py) ** 2))
    w_2d = np.sqrt(np.outer(wx, wy))  # 2D window normalized to conserve variance

    data_seg_w = np.zeros_like(data_seg_dt)  # Preallocate windowed data array

    for iseg in range(nseg):
        data_seg_w[:, :, iseg] = data_seg_dt[:, :, iseg] * w_2d

    #-----------------------------------------------------------------------
    # STEP #4 - Compute mean 2D wavenumber spectrum across all segments
    #-----------------------------------------------------------------------

    spec_sum = np.zeros((px, py))  # Initialize spectrum sum array
    var_time_seg = np.zeros(nseg)  # Store time domain variance per segment

    for iseg in range(nseg):
        
        # Compute 2D FFT and shift zero freq to center
        fft2D = fftshift(fft2(data_seg_w[:, :, iseg]))

        # Compute power spectrum (amplitude squared)
        amp = np.abs(fft2D) ** 2

        # Normalize spectrum to preserve variance and account for resolution
        amp_norm = amp / (px ** 2) / (py ** 2) / dkx / dky

        # Compute and store variance of windowed data segment (time domain)
        var_time_seg[iseg] = np.var(data_seg_w[:, :, iseg])

        # Accumulate normalized spectrum
        spec_sum += amp_norm

    # Compute mean spectrum across all segments
    m_spec = spec_sum / nseg

    #-----------------------------------------------------------------------
    # STEP #5 - Transform 2D spectrum to polar coordinates (k, theta)
    #-----------------------------------------------------------------------

    m_spec_pol, theta, k, _, _ = xy2rt(np.real(m_spec), kx, ky)

    #-----------------------------------------------------------------------
    # STEP #6 - Compute omni-directional spectrum by integrating over theta
    #-----------------------------------------------------------------------

    dtheta = theta[1] - theta[0]  # Angular resolution in radians
    spec_omni_i = np.zeros((len(k), len(theta)))

    for i in range(len(theta)):
        # Multiply by wavenumber magnitude and integrate over angle
        spec_omni_i[:, i] = (m_spec_pol[:, i] * k) * dtheta

    # Sum over all angles to get omni-directional spectrum
    spec_omni = np.nansum(spec_omni_i, axis=1)

    return m_spec, kx, ky, m_spec_pol, k, theta, spec_omni



#--- Compute the 2D Wavenumber Spectrum with padding and no Welch method ---# 
def spectrum2D_wavenumber_nonwelch_pad(data, x, y, dx, dy, frac):
    """
    Function for computing the 2D wavenumber spectrum from 2-dimensional 
    spatial data using zero padding (no Welch method).

    Parameters
    ----------
    data : 2D np.ndarray
        Evenly spaced spatial data.
    x : np.ndarray
        x-coordinate vector.
    y : np.ndarray
        y-coordinate vector.
    dx : float
        Spacing between x coordinates.
    dy : float
        Spacing between y coordinates.
    frac : float
        Fractional padding amount along each axis.

    Returns
    -------
    spec : 2D np.ndarray
        2D power spectrum (Cartesian).
    kx_pad : np.ndarray
        Wavenumber vector in x.
    ky_pad : np.ndarray
        Wavenumber vector in y.
    spec_pol : 2D np.ndarray
        Polar spectrum.
    k : np.ndarray
        Wavenumber magnitude.
    theta : np.ndarray
        Direction (radians).
    spec_omni : np.ndarray
        Omni-directional spectrum.
    """

    # Import functions
    import numpy as np
    from scipy.signal import windows
    from scipy.fft import fft2, fftshift
    from lsf import plane_lsf  
    from spectra import xy2rt  

    #-----------------------------------------------------------------------
    # STEP #0 - Remove linear trend using a planar least-squares fit
    #-----------------------------------------------------------------------
    Nx, Ny = data.shape
    X, Y = np.meshgrid(x, y, indexing='ij')
    plane_fit, *_ = plane_lsf(data, X, Y, method=1, weights=None)
    data_dt = data - plane_fit

    #-----------------------------------------------------------------------
    # STEP 1 - Apply normalized 2D Hanning window
    #-----------------------------------------------------------------------
    wx = windows.hann(Nx)
    wy = windows.hann(Ny)
    wx *= np.sqrt(Nx / np.sum(wx ** 2))
    wy *= np.sqrt(Ny / np.sum(wy ** 2))
    w_2d = np.outer(wx, wy)
    data_w = data_dt * w_2d

    #-----------------------------------------------------------------------
    # STEP 2 - Apply zero padding to the windowed data
    #-----------------------------------------------------------------------
    padx = np.zeros((Nx, round(Ny * frac)))
    data_padx = np.concatenate([padx, data_w, padx], axis=1)
    pady = np.zeros((round(Nx * frac), data_padx.shape[1]))
    data_pad = np.concatenate([pady, data_padx, pady], axis=0)

    #-----------------------------------------------------------------------
    # STEP 3 - Compute FFT and normalize power spectrum
    #-----------------------------------------------------------------------
    fft2D = fftshift(fft2(data_pad))
    amp = np.abs(fft2D) ** 2
    Nx_pad, Ny_pad = data_pad.shape
    dkx_pad = 2 * np.pi / (Nx_pad * dx)
    dky_pad = 2 * np.pi / (Ny_pad * dy)
    spec = amp / (Nx_pad**2 * Ny_pad**2 * dkx_pad * dky_pad)

    #-----------------------------------------------------------------------
    # STEP 4 - Compute Cartesian wavenumber vectors
    #-----------------------------------------------------------------------
    kx_pad = fftshift(np.fft.fftfreq(Nx_pad, dx)) * 2 * np.pi
    ky_pad = fftshift(np.fft.fftfreq(Ny_pad, dy)) * 2 * np.pi

    #-----------------------------------------------------------------------
    # STEP 5 - Check Parseval's theorem (energy conservation)
    #-----------------------------------------------------------------------
    var_time = np.var(data_dt)
    var_freq = np.sum(spec) * dkx_pad * dky_pad
    print(f"Time Domain: {var_time:.3e} m^2, Frequency Domain: {var_freq:.3e} m^2")

    #-----------------------------------------------------------------------
    # STEP 6 - Transform the 2D wavenumber spectrum to polar coordinates
    #-----------------------------------------------------------------------
    spec_pol, theta, k, _, _ = xy2rt(spec.real, kx_pad, ky_pad)

    #-----------------------------------------------------------------------
    # STEP 7 - Compute omni-directional spectrum
    #-----------------------------------------------------------------------
    dtheta = theta[1] - theta[0]
    spec_omni_i = (spec_pol.T * k).T * dtheta
    spec_omni = np.nansum(spec_omni_i, axis=1)

    return spec, kx_pad, ky_pad, spec_pol, k, theta, spec_omni


#--- Compute the 2D Wavenumber-frequency Spectrum ---# 
def spectrum2D_wavenumber_frequency(data, t, x, dt, dx, Mt, Mx):
    """
    Function for computing the 2D wavenumber-frequency spectrum from 2-dimensional
    spatiotemporal data T(t, x). The spectrum is computed by applying
    Hann windowing with 50% overlap along both time and space axes.

    Parameters
    ----------
    data : 2D np.ndarray
        Evenly spaced spatiotemporal data (time by space).
    t : np.ndarray
        Time vector.
    x : np.ndarray
        Spatial coordinate vector.
    dt : float
        Temporal spacing between measurements.
    dx : float
        Spatial spacing between measurements.
    Mt : int
        Number of segments to split the time axis into.
    Mx : int
        Number of segments to split the space axis into.

    Returns
    -------
    m_spec : 2D np.ndarray
        Averaged power spectrum in Cartesian coordinates (units: data^2/(rad/m * Hz))
    kx : np.ndarray
        Spatial wavenumber vector (rad/m)
    f : np.ndarray
        Frequency vector (Hz)
    m_spec_pol : 2D np.ndarray
        Averaged power spectrum in polar coordinates (units: data^2/(rad/m * rad))
    k : np.ndarray
        Wavenumber magnitude vector (rad/m)
    theta : np.ndarray
        Direction vector (radians) in (kx, f) space
    spec_omni : np.ndarray
        Omni-directional spectrum (units: data^2/(rad/m))
    """

    # Import libraries
    import numpy as np
    from scipy.signal import windows
    from scipy.fft import fft, fftshift
    from lsf import plane_lsf

    #-----------------------------------------------------------------------
    # STEP 0 - Set parameters and compute frequency and wavenumber vector
    #-----------------------------------------------------------------------
    Nt, Nx = data.shape
    pt = Nt // Mt  # points per time segment
    px = Nx // Mx  # points per space segment
    df = 2 * np.pi / (pt * dt)
    dkx = 2 * np.pi / (px * dx)

    # Adjust segment lengths for FFT evenness
    pt_fft = pt if pt % 2 == 0 else pt - 1
    px_fft = px if px % 2 == 0 else px - 1

    Lt = pt_fft * dt
    Lx = px_fft * dx

    # Wavenumber vector kx (rad/m)
    if pt_fft % 2 == 0:
        kx = (2 * np.pi / Lx) * np.concatenate((np.arange(0, px_fft // 2), np.arange(-px_fft // 2, 0)))
    else:
        kx = (2 * np.pi / Lx) * np.concatenate((np.arange(0, px_fft // 2 + 1), np.arange(-px_fft // 2, 0)))
    kx = fftshift(kx)

    # Frequency vector f (Hz)
    if pt_fft % 2 == 0:
        f = (1 / Lt) * np.concatenate((np.arange(0, pt_fft // 2), np.arange(-pt_fft // 2, 0)))
    else:
        f = (1 / Lt) * np.concatenate((np.arange(0, pt_fft // 2 + 1), np.arange(-pt_fft // 2, 0)))
    f = fftshift(f)

    #-----------------------------------------------------------------------
    # STEP 1 - Generate segments with 50% overlap in time and space
    #-----------------------------------------------------------------------
    if Mt > 1:
        nseg = Mt * Mx + (Mt - 1) * (Mx - 1)
    elif Mt == 1:
        nseg = Mt * Mx + Mt * (Mx - 1)

    # Preallocate arrays for segments
    data_seg = np.zeros((pt, px, nseg))
    t_seg = np.zeros((pt, nseg))
    x_seg = np.zeros((px, nseg))

    cn = 0

    # No overlap segments
    for it in range(Mt):
        idx_ti = it * pt
        idx_tf = idx_ti + pt

        for ix_ in range(Mx):
            idx_xi = ix_ * px
            idx_xf = idx_xi + px

            data_seg[:, :, cn] = data[idx_ti:idx_tf, idx_xi:idx_xf]
            t_seg[:, cn] = t[idx_ti:idx_tf]
            x_seg[:, cn] = x[idx_xi:idx_xf]
            cn += 1

    # 50% overlap segments
    for it in range(Mt - 1):
        idx_ti = int(it * pt + pt / 2)
        idx_tf = idx_ti + pt

        for ix_ in range(Mx - 1):
            idx_xi = int(ix_ * px + px / 2)
            idx_xf = idx_xi + px

            data_seg[:, :, cn] = data[idx_ti:idx_tf, idx_xi:idx_xf]
            t_seg[:, cn] = t[idx_ti:idx_tf]
            x_seg[:, cn] = x[idx_xi:idx_xf]
            cn += 1

    #-----------------------------------------------------------------------
    # STEP 2 - Remove linear trend (planar fit) from each segment
    #-----------------------------------------------------------------------
    data_seg_dt = np.zeros_like(data_seg)

    # Loop through segments
    for i in range(nseg):

        # Meshgrid for space and time (X=time, Y=space here)
        T_mesh, X_mesh = np.meshgrid(t_seg[:, i], x_seg[:, i], indexing='ij')

        # plane_lsf expects data shape (X,Y), so transpose segment data accordingly
        plane_fit, *_ = plane_lsf(data_seg[:, :, i].T, X_mesh.T, T_mesh.T, parameters=1, sigma=None)
        data_seg_dt[:, :, i] = (data_seg[:, :, i].T - plane_fit).T  # transpose back

    #-----------------------------------------------------------------------
    # STEP 3 - Apply Hann window to detrended data segments (normalized)
    #-----------------------------------------------------------------------
    wx = windows.hann(pt) * np.sqrt(pt / np.sum(windows.hann(pt) ** 2))
    wy = windows.hann(px) * np.sqrt(px / np.sum(windows.hann(px) ** 2))
    w_2d = np.outer(wx, wy)

    data_seg_w = np.zeros_like(data_seg_dt)
    for i in range(nseg):
        data_seg_w[:, :, i] = data_seg_dt[:, :, i] * w_2d

    #-----------------------------------------------------------------------
    # STEP 4 - Compute averaged 2D spectrum over all segments
    #-----------------------------------------------------------------------

    # Use adjusted FFT sizes for final spectrum
    spec_sum = np.zeros((pt_fft, px_fft))

    for i in range(nseg):

        fft2D = fftshift(np.fft.fft2(data_seg_w[:pt_fft, :px_fft, i]))
        amp = np.abs(fft2D) ** 2
        amp_norm = amp / (pt_fft ** 2) / (px_fft ** 2) / df / dkx
        spec_sum += amp_norm

    m_spec = spec_sum / nseg

    return m_spec, kx, f
