# General Variance Analysis functions
## Luke Colosi | lcolosi@ucsd.edu 

#--- 1D Variance Scale Analysis ---# 
def variance_scale_analysis_1D(ts, scales=None, fit_range=None, overlap=0.0, return_windows=False):
    
    """
    Compute variance as a function of scale (Mahadevan et al. 2002 style)
    for a 1D time series, with option for overlapping windows. Patchness is computed
    from the slope of the V(L). 

    Parameters
    ----------
    ts : array_like
        1D time series.
    scales : list or array, optional
        Window sizes (number of points) to evaluate. If None, uses
        powers of 2 up to length of time series.
    fit_range : tuple (min_L, max_L), optional
        Range of scales to include in slope fit. If None, fits across all.
    overlap : float, default 0.0
        Fractional overlap between windows. 
        - 0.0 → no overlap
        - 0.5 → 50% overlap
        - 0.75 → 75% overlap
        Must satisfy 0 <= overlap < 1.
    return_windows : bool, default False
        If True, also return dictionary of raw per-window variances.

    Returns
    -------
    L : ndarray
        Window sizes (scale).
    V : ndarray
        Normalized variance at each scale.
    p : float
        Best-fit slope of log(V) vs log(L).
    intercept : float
        Intercept of the fitted line in log-log space.
    window_vars_dict : dict, optional
        Dictionary keyed by scale L with list of per-window variances (normalized).
        Only returned if return_windows=True.
    """

    # Define libraries 
    import numpy as np

    #-----------------------------------#
    # STEP #1 - Compute V(L)
    #-----------------------------------#

    # Ensure input is a NumPy array and Total length of the time series
    ts = np.asarray(ts)              
    N = len(ts)                      

    # If no scales are provided, use powers of 2 up to the length of the time series
    if scales is None:
        max_power = int(np.floor(np.log2(N)))  # Maximum power of 2 less than N
        scales = [2**k for k in range(1, max_power+1)]  # e.g., [2,4,8,...]

    # Compute variance of full time series for normalization
    total_var = np.var(ts, ddof=1)   

    # Initialize list to store average variance per scale and dictionary to 
    # optionally store per-window variances
    V = []                           
    window_vars_dict = {}            

    # Loop over each window size / scale
    for L in scales:

        # Validate overlap input
        if not (0 <= overlap < 1):
            raise ValueError("overlap must be between 0 and 1 (fraction)")

        # Set step size between windows
        step = max(1, int(L * (1 - overlap)))  

        # Start indices for windows of length L
        starts = np.arange(0, N - L + 1, step)
        if len(starts) == 0:
            continue  # Skip if the window size is too large for the series

        # Store variance of each window
        window_vars = []  

        # Compute variance for each window
        for s in starts:

            # Extract segment/window
            segment = ts[s:s+L]                 

            # Compute variance
            window_vars.append(np.var(segment, ddof=1))  

        # Normalize by total variance and store per-window variances
        window_vars = np.array(window_vars) / total_var  
        window_vars_dict[L] = window_vars               

        # Average variance across all windows for this scale and store normalized 
        # average variance
        V_L = np.mean(window_vars)  
        V.append(V_L)               

    L = np.array(scales[:len(V)])  # Convert to NumPy array (ensure matching length)
    V = np.array(V)                # Convert variance list to NumPy array

    #-----------------------------------#
    # STEP #2 - Fit slope in log-log space 
    #-----------------------------------#

    # Log-transform for power-law relationship
    logL, logV = np.log10(L), np.log10(V)  

    # If a range for fitting is specified, only keep that subset
    if fit_range is not None:
        mask = (L >= fit_range[0]) & (L <= fit_range[1])
        logL, logV = logL[mask], logV[mask]

    # Linear fit in log-log space
    p, intercept = np.polyfit(logL, logV, 1)  

    # Return results, optionally including per-window variances
    if return_windows:
        return L, V, p, intercept, window_vars_dict
    else:
        return L, V, p, intercept
    

#--- 1D Variance Scale Analysis (Mask Aware)---# 
def variance_scale_analysis_1D_masked(ts, scales=None, fit_range=None, overlap=0.0, return_windows=False):

    """
    Compute variance as a function of scale (Mahadevan et al. 2002 style)
    for a 1D time series, with option for overlapping windows and masking support. 
    Patchness is computed from the slope of the V(L). Preserves the full list of 
    scales by padding invalid scales with masked values.

    Parameters
    ----------
    ts : array_like
        Input time series (1D). Can contain NaNs, which will be masked.
    scales : list or array, optional
        List of window sizes (in samples). Defaults to powers of 2 up to length of ts.
    fit_range : tuple, optional
        (Lmin, Lmax) range of scales to use for slope fitting in log-log space.
    overlap : float, optional
        Fractional overlap between adjacent windows (0 = no overlap, 0.5 = 50% overlap).
    return_windows : bool, optional
        If True, also returns dictionary of per-window variances.

    Returns
    -------
    scales : ndarray
        Array of scales (window lengths).
    V : np.ma.MaskedArray
        Normalized variance at each scale (masked where not valid).
    p : float
        Slope of log-log fit (exponent p).
    intercept : float
        Intercept of log-log fit.
    window_vars_dict : dict, optional
        Per-scale variances of individual windows (normalized).
    """

    # Import libraries 
    import numpy as np 

    #-----------------------------------#
    # STEP #1 - Compute V(L)
    #-----------------------------------#

    # Set nans to masked values and set total length of the time series
    ts = np.ma.masked_invalid(ts)
    N = len(ts)

    # If no scales are provided, use powers of 2 up to the length of the time series
    if scales is None:
        max_power = int(np.floor(np.log2(N)))
        scales = [2**k for k in range(1, max_power+1)] # e.g., [2,4,8,...]

    # Ensure scales is a NumPy array
    scales = np.array(scales)

    # Compute total variance for normalization (masked-aware)
    total_var = np.ma.var(ts, ddof=1)

    # Initialize list to store average variance per scale and dictionary to 
    # optionally store per-window variances
    V = np.ma.masked_all(len(scales))
    window_vars_dict = {}

    # Loop over each window size / scale
    for i, L in enumerate(scales):

        # Validate overlap input
        if not (0 <= overlap < 1):
            raise ValueError("overlap must be between 0 and 1 (fraction)")

        # Start indices for windows of length L
        step = max(1, int(L * (1 - overlap)))
        starts = np.arange(0, N - L + 1, step)
        if len(starts) == 0:
            continue # Skip if the window size is too large for the series

        # Store variance of each window
        window_vars = []

        # Loop through windows
        for s in starts:

            # Extract segment/window
            segment = ts[s:s+L]

            # Require at least 50% valid data and compute variances (mask aware)
            if segment.count() >= 0.5 * L:
                window_vars.append(np.ma.var(segment, ddof=1))

        # Skip if no valid windows
        if len(window_vars) == 0:
            continue

        # Normalize by total variance and store per-window variances
        window_vars = np.array(window_vars) / total_var
        window_vars_dict[L] = window_vars

        # Average variance across all windows for this scale and store normalized 
        # average variance
        V[i] = np.mean(window_vars)

    #-----------------------------------#
    # STEP #2 - Fit slope in log-log space 
    #-----------------------------------#

    # Find valid (non-masked) entries for fitting slope and Log-transform for power-law relationship
    valid = ~V.mask
    logL, logV = np.log10(scales[valid]), np.log10(V[valid])

    # If a range for fitting is specified, only keep that subset
    if fit_range is not None:
        mask = (scales[valid] >= fit_range[0]) & (scales[valid] <= fit_range[1])
        logL, logV = logL[mask], logV[mask]

    # Linear fit in log-log space
    if len(logL) > 1:
        p, intercept = np.polyfit(logL, logV, 1)
    else:
        p, intercept = np.nan, np.nan

    # Return results, optionally including per-window variances
    if return_windows:
        return scales, V, p, intercept, window_vars_dict
    else:
        return scales, V, p, intercept


