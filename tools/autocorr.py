# Real-valued Autocorrelation Analysis functions
## Luke Colosi | lcolosi@ucsd.edu 

#--- Autocorrelation and Autocovariance Function ---# 
def compute_autocorr_optimize(data, x, lag, bias, norm = 0):

    """
    rho_pos, rho_neg, R_pos, R_neg, x_ref_pos, x_ref_neg = compute_autocorr(data, x, lag, task, bias, norm = 0)

    Function for computing the autocovariance and autocorrelation 
    functions for positive and negative lag.
    
        Parameters
        ----------
        data : array
            Time or spatial series of data. This data must be preprocessed in the following ways:
            1) Detrend to remove any signal that is undesirable in the autocovariance function.
            2) Missing data gaps are filled with NaN values to ensure a continuous time series.
            3) Flagged data should be replaced with NaN values.

        x : array 
            Time or spatial vector for data record. 
            
        lag : int
            The desired number of lags for computing the correlation. The specified amount of lags is dependent
            on the length of the time series. You want to set the amount of lags to a value where the 
            correlation coefficent is for the proper amount of iterations along to fixed time series.
            Ex: lag_dt = len(data) (compute correlation coefficient at lag decreasing by one measurement at a time).
            
        bias : str
            Specifies whether the covariance is biased or unbiased. The unbiased estimate is normalized by
            1/n-m whereas the biased estimate is normalized by 1/n where n = total number of data points in time
            series and m is the lag time from 0 lag. Furthermore, specifies whether the correlation coefficent is biased 
            or unbaised using the same normalizations in numerator (unbiased (normalized by 1/n-m) or biased
            (normalized by 1/n)) and the normalization 1/n for both cases in the demominator. 
            Options: 'biased' or 'unbiased'.

        norm : int
            Specifies which lagged covariance you want to normalize the autocovariance function by. The normal convention
            is to normalize it by the variance of the data record (the zeroth lag). However in the case where the noise in the
            measurements is causing a large drop in the autocorrelation from the zeroth lag to the first lag (introducing 
            a decorrelation signal different from the decorrelation from the natural variability of the system), normalizing the
            autocovariance function by the first lag will provide a more accurate decorrelation scale. Options includes 0 or 1 
            corresponds to the zero and first lag respectively. Default value: norm = 0. 
            
        Returns
        -------
        rho_pos : array
            Positive lag autocorrelation function.
            
        rho_neg : array
            Negative lag autocorrelation function.
            
        R_pos : array
            Positive lag autocovariance function.
            
        R_neg : array
            Negative lag autocovariance function.

        x_ref_pos : array
            Lag variable for positive lag autocorrelation or autocovariance functions. 

        x_ref_neg : array 
            Lag variable for positive lag autocorrelation or autocovariance functions. 

        Libraries necessary to run function
        -----------------------------------
        import numpy as np 

    """

    # Import libraries 
    import numpy as np

    # Convert masked array to float array with NaNs for masked values (working with nan filled arrays is much faster than masked arrays)
    if np.ma.isMaskedArray(data):
        data_filled = data.filled(np.nan)
    else:
        data_filled = data

    # Choose interval length n which the correlation coefficient will be computed (Counting masked elements)
    N = len(data_filled)

    # Remove mean ignoring NaNs once for all data
    data_mean     = np.nanmean(data_filled)
    data_demeaned = data_filled - data_mean

    # Initialize autocovariance and autocorrelation arrays
    R         = np.zeros(lag)   # Autocovariance
    rho       = np.zeros(lag)   # Autocorrelation 

    # Compute normalization factor Rnorm used to scale autocovariances into autocorrelations
    if norm == 0:

        # Use variance (lag-0 autocovariance) as normalization
        valid_mask = ~np.isnan(data_demeaned)                # Identify valid (non-NaN) data points
        count_valid = np.sum(valid_mask)                     # Count of valid points
        
        # Compute variance ignoring NaNs by summing squared deviations and dividing by valid count
        Rnorm = np.nansum(data_demeaned * data_demeaned) / count_valid

    elif norm == 1:

        # Use lag-1 autocovariance as normalization
        valid_mask = ~np.isnan(data_demeaned[:-1]) & ~np.isnan(data_demeaned[1:]) # Identify valid (non-NaN) data points (ignoring the zero lag)
        count_valid = np.sum(valid_mask)                                          # Count valid pairs at lag 1
        
        # Compute lag-1 autocovariance ignoring NaNs by summing product of pairs divided by valid count
        Rnorm = np.nansum(data_demeaned[:-1][valid_mask] * data_demeaned[1:][valid_mask]) / count_valid
    else:
        raise ValueError("norm must be 0 or 1")

    # Loop through each lag interval to compute the correlation and covariance  
    for k in range(lag):

        # Create overlapping segments with lag k
        seg1 = data_demeaned[:N - k]
        seg2 = data_demeaned[k:]

        # Create mask for valid (non-NaN) pairs
        valid_mask = ~np.isnan(seg1) & ~np.isnan(seg2)
        n_eff = np.sum(valid_mask)
        n = len(seg1)

        if n_eff == 0:

            # No valid data pairs at this lag; assign NaN and skip computation
            R[k] = np.nan
            rho[k] = np.nan
            continue

        # Compute sum of products over valid pairs at lag k, ignoring NaNs
        inner_product = np.nansum(seg1[valid_mask] * seg2[valid_mask])

        # Compute the autocovariance function at lag k

        #--- Unbiased ---# 
        # Method: divide by number of valid pairs at this lag
        if bias == 'unbiased':
            R[k] = inner_product / n_eff

        #--- Unbiased Tapered (Triangular taper) ---# 
        # Method: same as unbiased estimate but also scale by ratio of available pairs to full length (triangular taper)
        elif bias == 'unbiased_tapered':
            R[k] = (inner_product / n_eff) * (n / N)

        #--- Biased ---# 
        # Method: divide by total length (fixed denominator)
        elif bias == 'biased':
            R[k] = inner_product / N

        # Catch if an incorrect argument for the bias argument is given
        else:
            raise ValueError("bias must be 'biased', 'unbiased', or 'unbiased_tapered'")

        # Compute the autocorrelation function at lag k
        rho[k] = R[k] / Rnorm


    # Combine positive and negative lag arrays 

    #--- Lag ---#  
    x_ref_pos = x[:lag] - x[0]
    x_ref_neg = -1 * np.flip(x_ref_pos)[:-1]

    #--- Autocovariance ---# 
    R_pos = R
    R_neg = np.flip(R)[:-1]

    #--- Autocorrelation ---# 
    if norm == 0:
        rho_pos = rho
        rho_neg = np.flip(rho)[:-1]
    elif norm == 1:
        # Set zero lag to 1 explicitly (normalize by first lag)
        rho_pos = np.insert(rho[1:], 0, 1)
        rho_neg = np.flip(rho)[:-1]

    return rho_pos, rho_neg, R_pos, R_neg, x_ref_pos, x_ref_neg

#--- Autocorrelation and Autocovariance Function ---# 
def compute_autocorr(data, x, lag, bias, norm = 0):

    """
    rho_pos, rho_neg, R_pos, R_neg, x_ref_pos, x_ref_neg = compute_autocorr(data, x, lag, task, bias, norm = 0)

    Function for computing the autocovariance and autocorrelation 
    functions for positive and negative lag.
    
        Parameters
        ----------
        data : array
            Time or spatial series of data. This data must be preprocessed in the following ways:
            1) Detrend to remove any signal that is undesirable in the autocovariance function.
            2) Missing data gaps are filled with NaN values to ensure a continuous time series.
            3) Flagged data should be replaced with NaN values.

        x : array 
            Time or spatial vector for data record. 
            
        lag : int
            The desired number of lags for computing the correlation. The specified amount of lags is dependent
            on the length of the time series. You want to set the amount of lags to a value where the 
            correlation coefficent is for the proper amount of iterations along to fixed time series.
            Ex: lag_dt = len(data) (compute correlation coefficient at lag decreasing by one measurement at a time).
            
        bias : str
            Specifies whether the covariance is biased or unbiased. The unbiased estimate is normalized by
            1/n-m whereas the biased estimate is normalized by 1/n where n = total number of data points in time
            series and m is the lag time from 0 lag. Furthermore, specifies whether the correlation coefficent is biased 
            or unbaised using the same normalizations in numerator (unbiased (normalized by 1/n-m) or biased
            (normalized by 1/n)) and the normalization 1/n for both cases in the demominator. 
            Options: 'biased' or 'unbiased'.

        norm : int
            Specifies which lagged covariance you want to normalize the autocovariance function by. The normal convention
            is to normalize it by the variance of the data record (the zeroth lag). However in the case where the noise in the
            measurements is causing a large drop in the autocorrelation from the zeroth lag to the first lag (introducing 
            a decorrelation signal different from the decorrelation from the natural variability of the system), normalizing the
            autocovariance function by the first lag will provide a more accurate decorrelation scale. Options includes 0 or 1 
            corresponds to the zero and first lag respectively. Default value: norm = 0. 
            
        Returns
        -------
        rho_pos : array
            Positive lag autocorrelation function.
            
        rho_neg : array
            Negative lag autocorrelation function.
            
        R_pos : array
            Positive lag autocovariance function.
            
        R_neg : array
            Negative lag autocovariance function.

        x_ref_pos : array
            Lag variable for positive lag autocorrelation or autocovariance functions. 

        x_ref_neg : array 
            Lag variable for positive lag autocorrelation or autocovariance functions. 

        Libraries necessary to run function
        -----------------------------------
        import numpy as np 

    """

    # Import libraries 
    import numpy as np

    # Choose interval length n which the correlation coefficient will be computed (Counting and discouting masked elements)
    N = len(data)
    N_eff = np.sum(~data.mask)

    # Initialize autocovariance and autocorrelation arrays
    R          = np.zeros(lag)   # Autocovariance
    rho        = np.zeros(lag)   # Autocorrelation 
    c_pairs_m  = np.zeros(lag)
    c_pairs_nm = np.zeros(lag)

    # Set the normalization factor for the autocorrelation 

    # Normalize by the zeroth lag 
    if norm == 0:

            # Set zero lagged data segments
            running = data[0:N]
            fix = data[:N-0]

            # Remove mean from each segment before computing covariance and correlation
            fix -= np.ma.mean(data)
            running -= np.ma.mean(data)

            # Compute the normalization
            Rnorm = (1/N_eff) * np.ma.dot(fix,running) #np.sum(data * np.conj(data))

    #--- Note ---# 
    # The normalization is the same for the biased, unbiased, and unbiased tapered estimates when normalizing with the zeroth lag
    # because the factors in front of the inner product are equivalent: 
    # 
    #       1/n_eff = (1 / n_eff) * (n / N) =  1/N_eff
    # 
    # This is because n_eff = N_eff and n = N at tau = 0. Recall that: 
    # 
    #                    n = N - k,
    # 
    # _eff denotes that the quantity excludes masked elements, and upper case N denotes that length of the full record
    # while lower case n denotes the length of the lagged record. 
    
    # Normalize by the first lag 
    elif norm == 1: 

        # Set lagged data segments
        running = data[1:N]
        fix = data[:N-1]

        # Remove mean from each segment before computing covariance and correlation
        fix -= np.ma.mean(data)
        running -= np.ma.mean(data)

        # Compute number of data pairs discounting pairs with masked vaules
        combined_mask = np.logical_or(fix.mask, running.mask)
        n_eff = np.sum(~combined_mask)
        n = len(running)
        
        #--- Unbiased ---# 
        if bias == 'unbiased':
            Rnorm = (1/n_eff) * np.ma.dot(fix,running) # np.sum(data[1:N] * np.conj(data[:N-1]))

        if bias == 'unbiased_tapered':
            Rnorm = (1/n_eff) * (n / N) * np.ma.dot(fix,running) # np.sum(data[1:N] * np.conj(data[:N-1]))

        #--- Biased ---# 
        if bias == 'biased': 
            Rnorm = (1/N_eff) * np.ma.dot(fix,running) # np.sum(data[1:N] * np.conj(data[:N-1]))

    # Loop through each lag interval to compute the correlation and covariance    
    for k in range(lag):

        # Set lagged data segments
        running = data[k:N]
        fix = data[:N-k]
        
        # Remove mean from each segment before computing covariance and correlation
        fix -= np.ma.mean(data)
        running -= np.ma.mean(data)

        # Compute the correlation coefficient terms at lag k
        inner_product = np.ma.dot(fix, running)

        # Compute the number of data pairs counting pairs with masked values
        n = len(running)  # Equivalent to N - k
        c_pairs_m[k] = n

        # Compute number of data pairs NOT counting pairs with masked vaules
        combined_mask = np.logical_or(fix.mask, running.mask)
        n_eff = np.sum(~combined_mask)
        c_pairs_nm[k] = n_eff
        
        # Compute autocorrelation and autocovariance function at lag k

        #--- Unbiased ---# 
        if bias == 'unbiased':
            R[k] = (1 / n_eff) * inner_product
            rho[k] = R[k] / Rnorm

        #--- Unbiased Tapered (Triangular taper) ---# 
        if bias == 'unbiased_tapered':
            R[k] = (1 / n_eff) * (n / N) * inner_product
            rho[k] = R[k] / Rnorm

        #--- Biased ---# 
        elif bias == 'biased':
            R[k] = (1 / N_eff) * inner_product
            rho[k] = R[k] / Rnorm
    
    # Combine positive and negative lag autocorrelation and autocovariance and set the lag vector
    if norm == 0:

        #--- Lag ---#  
        x_ref_pos = x - x[0]
        x_ref_neg = -1 * np.flip(x_ref_pos)[:-1]

        #--- Autocovariance ---# 
        R_pos = R
        R_neg = np.flip(R)[:-1]

        #--- Autocorrelation ---# 
        rho_pos = rho
        rho_neg = np.flip(rho)[:-1]

    elif norm == 1:

        #--- Lag ---#  
        x_ref_pos = x - x[0]
        x_ref_neg = -1 * np.flip(x_ref_pos)[:-1]

        #--- Autocovariance ---# 
        R_pos = R
        R_neg = np.flip(R)[:-1]

        #--- Autocorrelation ---# 
        rho_pos = np.insert(rho[1:], 0, 1) # Set zero lag to unity 
        rho_neg = np.flip(rho)[:-1]
    
    return rho_pos, rho_neg, R_pos, R_neg, x_ref_pos, x_ref_neg #, c_pairs_m, c_pairs_nm (ignoring this output)
    





#--- Decorrelation Scale Function (Integral Scale Estimate) ---%
def compute_decor_scale_optimize(autocorr,x_ref,dx,bias,norm):

    """
    Computes the decorrelation scale as an intergral time scale from the positively lag autocorrelation function.  

    Parameters
    ----------
    autocorr : array
            Positive lag autocorrelation function. 

    x_ref : array 
            Lag time or distance independent variable. 

    dx : float 
            The distance between data points in physical space. 

    bias : str
            Specifies whether the covariance is biased or unbiased. The unbiased estimate is normalized by
            1/n-m whereas the biased estimate is normalized by 1/n where n = total number of data points in time
            series and m is the lag time from 0 lag. Options: 'biased' or 'unbiased'.

    norm : int
            Specifies which lagged covariance you want to normalize the autocovariance function by. The normal convention
            is to normalize it by the variance of the data record (the zeroth lag). However in the case where the noise in the
            measurements is causing a large drop in the autocorrelation from the zeroth lag to the first lag (introducing 
            a decorrelation signal different from the decorrelation from the natural variability of the system), normalizing the
            autocovariance function by the first lag will provide a more accurate decorrelation scale. Options includes 0 or 1 
            corresponds to the zero and first lag respectively. Default value: norm = 0. 

    Returns
    -------
    scale : float 
        The integral time scale estimate of the decorrelation scale. 

    Libraries necessary to run function
    -----------------------------------
    import numpy as np 
    from scipy.integrate import trapezoid

    """

    # Import libraries 
    import numpy as np
    from scipy.integrate import cumulative_trapezoid

    # Set the length of data series and data interval
    N = len(autocorr)      # length of one-sided autocorrelation function (and number of samples in data record)
    R = N * dx             # length of the data series (units of time or space)
    
    # Normalize by the zeroth lag 
    if norm == 0: 

        # Set the positive and negative lagged autocovariance functions and concatinate
        autocorr_full = np.concatenate((np.flip(autocorr[1:]), autocorr))
        x_ref_full    = np.concatenate((-1 * np.flip(x_ref[1:]), x_ref))

        # Precompute all possible trapezoidal integrals over the full symmetric autocorrelation function
        if bias == 'unbiased':
            
            # Compute triangular weights for unbiased estimator:
            # Each lag value is weighted by (1 - |lag| / R), which accounts for the decreasing number of data pairs at larger lags
            weights = 1 - (np.abs(x_ref_full) / R)
            
            # Multiply the symmetric autocorrelation function by the weights
            # This creates the weighted integrand to be used in integration
            integrand = weights * autocorr_full

        else:
            # If using biased estimator, do not apply weights; just use raw autocorrelation values
            integrand = autocorr_full

        # Interpolate over nans 
        if np.any(np.isnan(integrand)):

            # Find nan indices
            nans = np.isnan(integrand)

            # Linearly interpolate over nans
            integrand[nans] = np.interp(x_ref_full[nans], x_ref_full[~nans], integrand[~nans]) 

            # Define clean variable
            integrand_clean = integrand
        else: 
            integrand_clean = integrand

        # Compute the cumulative integral of the (possibly weighted) autocorrelation function
        # The result is a 1D array where each entry gives the integral from the first lag up to that point in x_ref_full
        # Setting initial=0 ensures the integral starts at zero
        integral_full = cumulative_trapezoid(integrand_clean, x_ref_full, initial=0)

        # Determine the index corresponding to zero lag in the full symmetric autocorrelation array
        center = N - 1  # Since autocorr_full has length 2N - 1, the center index is at N-1

        # Initialize an array to hold decorrelation scale estimates for each lag i
        scale_N = np.zeros(N)

        # Loop over lags from 1 to N-1 to compute integral estimates over symmetric windows
        for i in range(1, N):

            # Calculate the start index of the symmetric window for negative lag (-i)
            start = center - i

            # Calculate the end index (one past the positive lag +i) for slicing
            end = center + i + 1

            # Use cumulative trapezoidal integral differences to compute integral over [-i, +i]
            scale_N[i] = integral_full[end - 1] - integral_full[start]

        # Select the maximum value from all computed scales as a conservative estimate of the decorrelation scale
        scale = np.nanmax(scale_N)

    #--- Normalize by the first lag ---#
    if norm == 1:

        # For the positive lag autocorrelation function only, assuming symmetry around lag zero,
        # so the integral over negative lags can be accounted for by doubling the positive lag integral

        if bias == 'unbiased':

            # Compute triangular weights to correct for bias (fewer data pairs at larger lags)
            # weights linearly decrease from 1 at lag zero to 0 at maximum lag R
            weights = 1 - (x_ref / R)

            # Apply the weights to the autocorrelation values to get the weighted integrand
            integrand = weights * autocorr
        else:

            # If biased, no weighting; use the raw autocorrelation values
            integrand = autocorr

        # Compute the cumulative integral (trapezoidal rule) of the integrand over positive lags
        # integral_pos[k] holds the integral from lag zero up to lag x_ref[k]
        integral_pos = cumulative_trapezoid(integrand, x_ref, initial=0)

        # Initialize array to hold decorrelation scale estimates for each lag
        scale_N = np.zeros(N)

        # For lags greater than zero, scale the integral by 2 to account for symmetric negative lags
        scale_N[1:] = 2 * integral_pos[1:]

        # Take the maximum integral value as the conservative estimate of the decorrelation scale
        scale = np.nanmax(scale_N)

    return scale

#--- Decorrelation Scale Function (Integral Scale Estimate) ---%
def compute_decor_scale(autocorr,x_ref,dx,bias,norm):

    """
    Computes the decorrelation scale as an intergral time scale from the positively lag autocorrelation function.  

    Parameters
    ----------
    autocorr : array
            Positive lag autocorrelation function. 

    x_ref : array 
            Lag time or distance independent variable. 

    dx : float 
            The distance between data points in physical space. 

    bias : str
            Specifies whether the covariance is biased or unbiased. The unbiased estimate is normalized by
            1/n-m whereas the biased estimate is normalized by 1/n where n = total number of data points in time
            series and m is the lag time from 0 lag. Options: 'biased' or 'unbiased'.

    norm : int
            Specifies which lagged covariance you want to normalize the autocovariance function by. The normal convention
            is to normalize it by the variance of the data record (the zeroth lag). However in the case where the noise in the
            measurements is causing a large drop in the autocorrelation from the zeroth lag to the first lag (introducing 
            a decorrelation signal different from the decorrelation from the natural variability of the system), normalizing the
            autocovariance function by the first lag will provide a more accurate decorrelation scale. Options includes 0 or 1 
            corresponds to the zero and first lag respectively. Default value: norm = 0. 

    Returns
    -------
    scale : float 
        The integral time scale estimate of the decorrelation scale. 

    Libraries necessary to run function
    -----------------------------------
    import numpy as np 
    from scipy.integrate import trapezoid

    """

    # Import libraries 
    import numpy as np
    from scipy.integrate import trapezoid
    
    # Normalize by the zeroth lag 
    if norm == 0: 

        # Set the positive and negative lagged autocovariance functions
        autocorr_pos = autocorr
        autocorr_neg = np.flip(autocorr)[:-1]

        # Set the positive and negative lag variable
        x_ref_pos = x_ref
        x_ref_neg = -1 * np.flip(x_ref)[:-1]

        # Set the length of data series and data interval
        N = len(autocorr_pos)      # length of one-sided autocorrelation function (and number of samples in data record)
        R = N * dx                 # length of the data series (units of time or space)

        # Initialize scale_N variable
        scale_N = np.zeros(N)

        # Loop through lags 
        for i in range(N):

                # Index autocorrelation function 

                #--- Zeroth lag ---# 
                if i == 0:

                        # Set interal of the autocorrelation function to zero (intergration range vanishes)
                        iscale = 0  

                #--- Higher Order lag ---#     
                else:

                        # Index the autocorrelation function and combine the negative and positive lagged autocorrelation functions 
                        coef = np.concatenate((autocorr_neg[N-i-1:], autocorr_pos[:i+1]))
                        x    = np.concatenate((x_ref_neg[N-i-1:], x_ref_pos[:i+1]))

                        # Compute time or spatial lag
                        r = i * dx

                        # Compute integral of autocorrelation function (with triangular filter weighting lower lags for unbiased estimator)
                        if bias == 'unbiased':
                                iscale = trapezoid((1 - (r / R)) * coef, x, dx=dx)
                        elif bias == 'biased':
                                iscale = trapezoid(coef, x, dx=dx)

                # Save the ith scaling factor=
                scale_N[i] = iscale

        # Find the maximum decorrelation time scale (conservative estimate)
        scale = np.nanmax(scale_N)

    #--- Normalize by the first lag ---#
    elif norm == 1: 

        # Set the positive lagged autocovariance functions
        autocorr_pos = autocorr

        # Set the length of data series and data interval
        N = len(autocorr_pos)  # length of one-sided autocorrelation function
        R = N * dx             # length of the data series

        # Initialize scale_N variable
        scale_N = np.zeros(N)

        # Loop through lags 
        for i in range(N):

                # Index autocorrelation function 

                #--- Zeroth and first lag ---# 
                if i == 0:

                        # Set interal of the autocorrelation function to zero (intergration range vanishes)
                        iscale = 0  

                #--- Higher Order lag ---#     
                else:

                        # Index the autocorrelation function and combine the negative and positive lagged autocorrelation functions 
                        autocor_pos_lag = autocorr_pos[:i+1]

                        # Index the autocorrelation function 
                        x_ref_pos_lag = x_ref[:i+1]

                        # Compute spatial lag
                        r = i * dx

                        # Compute integral of autocorrelation function (using the symmetry across the y-axis)
                        if bias == 'unbiased': 
                               iscale_pos = trapezoid((1 - (r / R)) * autocor_pos_lag, x_ref_pos_lag, dx=dx) 
                        elif bias == 'biased': 
                               iscale_pos = trapezoid(autocor_pos_lag, x_ref_pos_lag, dx=dx) 
                        iscale = 2*iscale_pos

                # Save the ith scaling factor
                scale_N[i] = iscale


        # Find the maximum decorrelation time scale (conservative estimate)
        scale = np.nanmax(scale_N)

    return scale 





#--- Autocorrelation Glider transect with bin averaging function ---# 
def compute_glider_autocorr(dist, data, water_depth, L, on_lim, off_lim, trans_lim, dir, estimator, option_plot, dirOut):

    """
    (
        autocorr_on, autocorr_trans, autocorr_off, 
        autocorr_on_norm, autocorr_trans_norm, autocorr_off_norm, 
        autocov_on, autocov_trans, autocov_off, autocov_full, 
        L_on, L_trans, L_off, 
        dist_on, dist_trans, dist_off, dist_bin, 
        dist_scale_on, dist_scale_trans, dist_scale_off 
    ) = compute_glider_autocorr(
        dist, data, water_depth, 
        L, on_lim, off_lim, trans_lim, 
        dir, estimator, option_plot, dirOut
    )

    Function for computing the autocorrelation functions for scalar quantities for on/off-shelf and in the transition region for a 
    given glider transect.
    
        Parameters
        ----------
        dist : array (units: kilometers)
            Distance from shore (releative to point conception) for a single glider transect. 

        data : array (units: dependent on scalar quantity)
            Scalar data along the spray glider transect. This can be temperature, salinity, density, chlorophyll, and others. 
            
        water_depth : array (units: meters)
            Water depth along the glider transect referenced to the ocean surface (z = 0 with the ocean interior being negative)
            
        L : float (units: kilometers)
            The distance between points for the along track regular spatial grid.  
            
        on_lim : float (units: meters)
            Specifies the depth limit for the onshelf region. Water depth origin is at the ocean surface (depths are negative).  
            
        off_lim : float (units: meters)
            Specifies the depth limit for the offshelf region. Water depth origin is at the ocean surface (depths are negative).

        trans_lim : array (units: meters)
            Specifies the depth limit for the transition region. Water depth origin is at the ocean surface (depths are negative). 
            Argument takes the form: 
                    trans_lim = [on_lim + dx_on, off_lim + dx_off]
            
        dir : Float (units: [])
            Specifies the direction relative to shore the glider is moving (Moving Onshore = -1, Moving Offshore = 1). This is used
            for definiting the regular spatial grid of the bin averaged data.  

        estimator : str
            String specifying the type of autocorrelation estimator. Options include: 'biased', 'unbiased', and 'unbiased_tapered'
        
        option_plot : boolean
            Specifies if supplementary plots are plotted. 
        
        dirOut : str
            Specifies the directory to save the intermediate quality control figures. 
             
        Returns
        -------
        autocorr_on : array
            Positive lag autocorrelation function for the on-shelf region.

        autocorr_trans : array
            Positive lag autocorrelation function for the transition region.

        autocorr_off : array
            Positive lag autocorrelation function for the off-shelf region.

        autocorr_on_norm : array
            Positive lag autocorrelation function for the on-shelf region.

        autocorr_trans_norm : array


        autocorr_off_norm : array


        autocov_on : array


        autocov_trans : array


        autocov_off : array


        autocov_full : array


        L_on : array


        L_trans : array


        L_off : array


        dist_on : array


        dist_trans : array


        dist_off : array


        dist_bin : array


        dist_scale_on : array


        dist_scale_trans : array


        dist_scale_off : array
            
        Libraries necessary to run function
        -----------------------------------
        import numpy as np
        import pandas as pd
        from scipy.ndimage import gaussian_filter1d
        from autocorr import compute_decor_scale, compute_autocorr
        from lsf import detrend
        from plotScaleAnalysis import plot_depth_data_autocorr

    """

    # Import libraries 
    import numpy as np
    import pandas as pd
    from scipy.ndimage import gaussian_filter1d
    from autocorr import compute_decor_scale, compute_autocorr
    from lsf import detrend
    from plotScaleAnalysis import plot_depth_data_autocorr


    ##################################################################
    ## STEP #1 - Bin the data and water depth onto a regular spatial grid 
    ##################################################################

    # Set the bin edges of uniform spatial along track grid
    if dir == -1:
        dist_edges = np.arange(dist[-1],dist[0] + L, L)  
    else: 
        dist_edges = np.arange(dist[0],dist[-1] + L, L)  

    # Set the bin center for the uniform spatial along track grid  
    dist_bin = dist_edges[:-1] + np.diff(dist_edges)/2 

    # Create a pandas DataFrame
    df = pd.DataFrame({'distance': dist, 'data': data, 'water_depth': water_depth})

    # Assign each data point to a bin
    df['bin'] = pd.cut(df['distance'], bins=dist_edges, labels=dist_bin, include_lowest=True)

    # Compute bin-averaged data and water depth
    binned_data = df.groupby('bin').agg(
        mean_data =('data', 'mean'),
        std_data =('data', 'std'),
        mean_water_depth=('water_depth', 'mean'),
        std_water_depth=('water_depth', 'std'),
        count=('data', 'count')  
    ).reset_index()

    # Extract data from dataframe and mask NaNs 
    data_bin = np.ma.masked_invalid(binned_data['mean_data'].values)
    water_depth_bin = np.ma.masked_invalid(binned_data['mean_water_depth'].values)
    counts = np.ma.masked_invalid(binned_data['count'].values)


    ##################################################################
    ## STEP #2 - Split record into on-shelf, transiton, and off-shelf regions 
    ##################################################################

    # Fill masked values with interpolation (or NaN)
    water_depth_filled = water_depth_bin.filled(np.nan)  # Convert to a normal array with NaNs

    # Interpolate missing values before applying smoothing
    valid = ~np.isnan(water_depth_filled)
    water_depth_filled[~valid] = np.interp(np.flatnonzero(~valid), np.flatnonzero(valid),water_depth_filled[valid])

    # Smooth the water depth record using a gaussian filter so that spikes and valleys are not identified in another section 
    water_depth_filtered = np.ma.array(gaussian_filter1d(water_depth_filled, sigma=5))

    # Convert back to a masked array, keeping the original mask
    water_depth_sm = np.ma.array(water_depth_filtered, mask=water_depth_bin.mask)

    ##################################################################
    ## STEP #2a - On-Shelf 
    ##################################################################

    # Fill masked values with 9999 (for preserving masked elements in the array when indexing for on-shelf case)
    water_depth_sm_fill = np.array(water_depth_sm.filled(9999))
    water_depth_bin_fill = np.array(water_depth_bin.filled(9999))

    # Find the indices for on-shelf 
    idx_on = water_depth_sm_fill >= on_lim

    # Grab distance, data, and water depth from binned data
    dist_on_fill = dist_bin[idx_on]
    data_on_fill = data_bin[idx_on]
    water_depth_on_fill = water_depth_bin_fill[idx_on]

    #--- Remove any fill values at the end of the record ---# 

    # Find the indices of all the filled values and take their difference
    idx_fill = np.where(water_depth_on_fill == 9999)[0]
    idx_fill_diff = np.diff(idx_fill)

    # Find the index of the last one in the sequence if masked values are present in the record
    if len(idx_fill_diff) == 0:

        # Set distance, data, and water depth to un-indexed variables when no masked values are present
        dist_on = dist_on_fill
        data_on = data_on_fill
        water_depth_on_fill_trim = water_depth_on_fill

    elif len(idx_fill_diff) != 0:

        #--- If no fill values at the end of the record ---# 
        if idx_fill_diff[-1] != 1:

            # Set index to 0
            idx_one = -1 

        #--- If fill values do exist at the end of the record ---# 
        else:
            idx_one = len(idx_fill_diff) - np.argmax(idx_fill_diff[::-1] != 1)

        # Remove fill values at the beginning 
        dist_on = dist_on_fill[:idx_fill[idx_one]]
        data_on = data_on_fill[:idx_fill[idx_one]]
        water_depth_on_fill_trim = water_depth_on_fill[:idx_fill[idx_one]]

    # Replace rest of the filled values with masks (for preserving masked elements in the array)
    water_depth_on = np.ma.masked_equal(water_depth_on_fill_trim, 9999)

    ##################################################################
    ## STEP #2b - Off-Shelf 
    ##################################################################

    # Fill masked values with -9999 (for preserving masked elements in the array when indexing for off-shelf)
    water_depth_sm_fill = np.array(water_depth_sm.filled(-9999))
    water_depth_bin_fill = np.array(water_depth_bin.filled(-9999))

    # Find the indices for off and on shelf 
    idx_off = water_depth_sm_fill <= off_lim

    # Grab distance, data, and water depth from binned data
    dist_off_fill = dist_bin[idx_off]
    data_off_fill = data_bin[idx_off]
    water_depth_off_fill = water_depth_bin_fill[idx_off]

    #--- Remove any fill values at the beginning of the record ---#

    # Find the indices of all the filled values and take their difference
    idx_fill = np.where(water_depth_off_fill == -9999)[0]
    idx_fill_diff = np.diff(idx_fill)

    # Find the index of the last one in the sequence if masked values are present in the record
    if len(idx_fill_diff) == 0:

        # Set distance, data, and water depth to un-indexed variables when no masked values are present
        dist_off = dist_off_fill
        data_off = data_off_fill
        water_depth_off_fill_trim = water_depth_off_fill

    elif len(idx_fill_diff) != 0:

        #--- If no fill values at the beginning of the record ---# 
        if idx_fill_diff[0] != 1:

            # Set index to 0
            idx_one = 0  

        #--- If fill values do exist at the beginning of the record ---# 
        else:
            idx_one = np.argmax(idx_fill_diff != 1) - 1 if np.any(idx_fill_diff != 1) else len(idx_fill_diff) - 1
            
        # Remove fill values at the beginning 
        dist_off = dist_off_fill[idx_one + 2:]
        data_off = data_off_fill[idx_one + 2:]
        water_depth_off_fill_trim = water_depth_off_fill[idx_one + 2:]

    # Replace rest of the filled values with masks (for preserving masked elements in the array)
    water_depth_off = np.ma.masked_equal(water_depth_off_fill_trim, -9999)

    ##################################################################
    ## STEP #2c - Transition Region
    ##################################################################

    # Compute the mean of the water depth within the transition region 
    mean_water_depth = np.ma.mean(water_depth_sm[(water_depth_sm >= off_lim) & (water_depth_sm <= on_lim)])

    # Fill masked values with mean water depth within transition section  (for preserving masked elements in the array when indexing for off-shelf)
    water_depth_sm_fill = np.array(water_depth_sm.filled(mean_water_depth))
    water_depth_bin_fill = np.array(water_depth_bin.filled(mean_water_depth))

    # Extend the limits of the transition region to include a bit of the 
    upper_lim = trans_lim[0]
    lower_lim = trans_lim[1] 

    # Find the indices for the transition region 
    idx_trans = (water_depth_sm_fill >= lower_lim) & (water_depth_sm_fill <= upper_lim)

    # Grab distance, data, and water depth from binned data
    dist_trans_fill = dist_bin[idx_trans]
    data_trans_fill = data_bin[idx_trans]
    water_depth_trans_fill = water_depth_bin_fill[idx_trans]

    #--- Remove any fill values at the beginning of the record ---#

    # Find the indices of all the filled values and take their difference
    idx_fill_st = np.where(water_depth_trans_fill == mean_water_depth)[0]
    idx_fill_diff = np.diff(idx_fill_st)

    # Find the index of the last one in the sequence if masked values are present in the record
    if (len(idx_fill_diff) == 0):

        # Set distance, data, and water depth to un-indexed variables when no masked values are present
        dist_off_i = dist_trans_fill
        data_off_i = data_trans_fill
        water_depth_trans_fill_trim_i = water_depth_trans_fill

    elif len(idx_fill_diff) != 0:

        #--- If no fill values at the beginning of the record ---# 
        if idx_fill_st[0] != 0:

            # Set index to 0
            idx_one = 0  

        #--- If fill values do exist at the beginning of the record ---# 
        else:
            idx_one = np.argmax(idx_fill_diff != 1) - 1 if np.any(idx_fill_diff != 1) else len(idx_fill_diff) - 1

        # Remove fill values at the beginning 
        dist_trans_i = dist_trans_fill[idx_one + 2:]
        data_trans_i = data_trans_fill[idx_one + 2:]
        water_depth_trans_fill_trim_i = water_depth_trans_fill[idx_one + 2:]

    #--- Remove any fill values at the end of the record ---# 

    # Find the indices of all the filled values and take their difference
    idx_fill_ed = np.where(water_depth_trans_fill_trim_i == mean_water_depth)[0]
    idx_fill_diff = np.diff(idx_fill_ed)

    # Find the index of the last one in the sequence if masked values are present in the record
    if len(idx_fill_diff) == 0:

        # Set distance, data, and water depth to un-indexed variables when no masked values are present
        dist_trans = dist_trans_i
        data_trans = data_trans_i
        water_depth_trans_fill_trim = water_depth_trans_fill_trim_i

    elif len(idx_fill_diff) != 0:

        #--- If no fill values at the end of the record ---# 
        if idx_fill_ed[-1] != (len(water_depth_trans_fill_trim_i) - 1):

            # Set index to the last index in the record
            idx_one = -1 

        #--- If fill values are just present at the end of the array (none at the beginning or within the section) ---#
        elif (idx_fill_st[0] != 0) & (np.all(idx_fill_diff == 1)):

            # Set index to the first index
            idx_one = 0

        #--- If fill values do exist at the end of the record ---# 
        else:
            idx_one = len(idx_fill_diff) - np.argmax(idx_fill_diff[::-1] != 1)

        # Remove fill values at the beginning 
        dist_trans = dist_trans_i[:idx_fill_ed[idx_one]]
        data_trans = data_trans_i[:idx_fill_ed[idx_one]]
        water_depth_trans_fill_trim = water_depth_trans_fill_trim_i[:idx_fill_ed[idx_one]]

    # Replace rest of the filled values with masks (for preserving masked elements in the array)
    water_depth_trans = np.ma.masked_equal(water_depth_trans_fill_trim, mean_water_depth)

    
    ##################################################################
    ## STEP #3 - Detrend data record 
    ##################################################################
    data_on_dt    = detrend(data_on, dist_on, mean = 0)      #- np.ma.mean(data_on)
    data_trans_dt = detrend(data_trans, dist_trans, mean = 0) #- np.ma.mean(data_trans)
    data_off_dt   = detrend(data_off, dist_off, mean = 0)     #- np.ma.mean(data_off)
    data_full_dt  = detrend(data_bin, dist_bin, mean = 0)     #- np.ma.mean(data_bin) 


    ##################################################################
    ## STEP #4 - Compute autocorrelation and Decorrelation Scale function 
    ##################################################################

    # Set parameters
    lag_on, lag_trans, lag_off, lag_full = len(data_on_dt), len(data_trans_dt), len(data_off_dt), len(data_full_dt)
    
    # Compute autocorrelation function normalized by zeroth lag (for averaging) 
    autocorr_on, _, _, _, dist_scale_on, _ = compute_autocorr(data_on_dt, dist_on, lag_on, estimator, 0)
    autocorr_trans, _, _, _, dist_scale_trans, _ = compute_autocorr(data_trans_dt, dist_trans, lag_trans, estimator, 0)
    autocorr_off, _, _, _, dist_scale_off, _ = compute_autocorr(data_off_dt, dist_off, lag_off, estimator, 0)

    # Compute autocorrelation function normalized by the first lag (for transect decorrelation)
    autocorr_on_n, _, _, _, dist_scale_on_n, _ = compute_autocorr(data_on_dt, dist_on, lag_on, estimator, 1)
    autocorr_trans_n, _, _, _, dist_scale_trans_n, _ = compute_autocorr(data_trans_dt, dist_trans, lag_trans, estimator, 1)
    autocorr_off_n, _, _, _, dist_scale_off_n, _ = compute_autocorr(data_off_dt, dist_off, lag_off, estimator, 1)

    # Cmpute the unbiased autocovariance function (for spectra computation)
    _, _, autocov_on, _, _, _ = compute_autocorr(data_on_dt, dist_on, lag_on, 'unbiased', 0)
    _, _, autocov_trans, _, _, _ = compute_autocorr(data_trans_dt, dist_trans, lag_trans, 'unbiased', 0)
    _, _, autocov_off, _, _, _ = compute_autocorr(data_off_dt, dist_trans, lag_off, 'unbiased', 0)
    _, _, autocov_full, _, _, _ = compute_autocorr(data_full_dt, dist_bin, lag_full, 'unbiased', 0)

    # Compute decorrelation time scale 
    L_on    = compute_decor_scale(autocorr_on,dist_scale_on,L,estimator,0)
    L_trans = compute_decor_scale(autocorr_trans,dist_scale_trans,L,estimator,0)
    L_off   = compute_decor_scale(autocorr_off,dist_scale_off,L,estimator,0)


    ##################################################################
    ## STEP 5 - Plot water depth, data, and autocorrelation function
    ##################################################################

    # Set plotting parameters 
    fontsize = 14

    # Plot water depth, data, and autocorrelation  
    if option_plot == 1:
        plot_depth_data_autocorr(dist, dist_on, dist_trans, dist_off, water_depth, water_depth_on, water_depth_trans, water_depth_off, data, data_on, data_trans, data_off, autocorr_on, autocorr_trans, autocorr_off, on_lim, off_lim, fontsize, dirOut)

    return autocorr_on, autocorr_trans, autocorr_off, autocorr_on_n, autocorr_trans_n, autocorr_off_n, autocov_on, autocov_trans, autocov_off, autocov_full, L_on, L_trans, L_off, dist_on, dist_trans, dist_off, dist_bin, dist_scale_on_n, dist_scale_trans_n, dist_scale_off_n





#--- Autocorrelation Glider transect with bin averaging function ---# 
def compute_glider_autocorr_interp(dist, data, water_depth, L, on_lim, off_lim, trans_lim, dir, estimator, option_plot, dirOut):

    """
    (
        autocorr_on, autocorr_trans, autocorr_off, 
        autocorr_on_norm, autocorr_trans_norm, autocorr_off_norm, 
        autocov_on, autocov_trans, autocov_off, autocov_full, 
        L_on, L_trans, L_off, 
        dist_on, dist_trans, dist_off, dist_bin, 
        dist_scale_on, dist_scale_trans, dist_scale_off 
    ) = compute_glider_autocorr_interp(
        dist, data, water_depth, 
        L, on_lim, off_lim, trans_lim, 
        dir, estimator, option_plot, dirOut
    )

    Function for computing the autocorrelation functions for scalar quantities for on/off-shelf and in the transition region for a 
    given glider transect.
    
        Parameters
        ----------
        dist : array (units: kilometers)
            Distance from shore (releative to point conception) for a single glider transect. 

        data : array (units: dependent on scalar quantity)
            Scalar data along the spray glider transect. This can be temperature, salinity, density, chlorophyll, and others. 
            
        water_depth : array (units: meters)
            Water depth along the glider transect referenced to the ocean surface (z = 0 with the ocean interior being negative)
            
        L : float (units: kilometers)
            The distance between points for the along track regular spatial grid.  
            
        on_lim : float (units: meters)
            Specifies the depth limit for the onshelf region. Water depth origin is at the ocean surface (depths are negative).  
            
        off_lim : float (units: meters)
            Specifies the depth limit for the offshelf region. Water depth origin is at the ocean surface (depths are negative).

        trans_lim : array (units: meters)
            Specifies the depth limit for the transition region. Water depth origin is at the ocean surface (depths are negative). 
            Argument takes the form: 
                    trans_lim = [on_lim + dx_on, off_lim + dx_off]
            
        dir : Float (units: [])
            Specifies the direction relative to shore the glider is moving (Moving Onshore = -1, Moving Offshore = 1). This is used
            for definiting the regular spatial grid of the bin averaged data.  

        estimator : str
            String specifying the type of autocorrelation estimator. Options include: 'biased', 'unbiased', and 'unbiased_tapered'
        
        option_plot : boolean
            Specifies if supplementary plots are plotted. 
        
        dirOut : str
            Specifies the directory to save the intermediate quality control figures. 
             
        Returns
        -------
        autocorr_on : array
            Positive lag autocorrelation function for the on-shelf region.

        autocorr_trans : array
            Positive lag autocorrelation function for the transition region.

        autocorr_off : array
            Positive lag autocorrelation function for the off-shelf region.

        autocorr_on_norm : array
            Positive lag autocorrelation function for the on-shelf region.

        autocorr_trans_norm : array


        autocorr_off_norm : array


        autocov_on : array


        autocov_trans : array


        autocov_off : array


        autocov_full : array


        L_on : array


        L_trans : array


        L_off : array


        dist_on : array


        dist_trans : array


        dist_off : array


        dist_bin : array


        dist_scale_on : array


        dist_scale_trans : array


        dist_scale_off : array
            
        Libraries necessary to run function
        -----------------------------------
        import numpy as np
        import pandas as pd
        from scipy.ndimage import gaussian_filter1d
        from autocorr import compute_decor_scale, compute_autocorr
        from lsf import detrend
        from plotScaleAnalysis import plot_depth_data_autocorr

    """

    # Import libraries 
    import numpy as np
    import pandas as pd
    from scipy.interpolate import interp1d
    from scipy.ndimage import gaussian_filter1d
    from autocorr import compute_decor_scale, compute_autocorr
    from lsf import detrend
    from plotScaleAnalysis import plot_depth_data_autocorr
    import matplotlib.pyplot as plt


    ##################################################################
    ## STEP #1 - Interpolate the data and water depth onto a regular spatial grid 
    ##################################################################

    # Set the regular spatial grid 
    if dir == -1:
        dist_int = np.arange(dist[-1],dist[0] + L, L)  
    else: 
        dist_int = np.arange(dist[0],dist[-1] + L, L)  

    # Identify large gaps and create a boolean array for the valid points to interpolate across 
    valid_segments = np.diff(dist) <= 2.5*L                  # Boolean mask for valid transitions (set so at most 2 interpolated points can be between two data points)
    valid_points = np.concatenate(([True], valid_segments))  # Keep first point
    valid_points[1:] &= valid_segments                       # Ensure both points in a segment are valid

    #--- Split into segments where gaps are greater or equal to gap threshold ---# 

    # Initialize arrays 
    segments = []  # Storages tuples of valid segments 
    segments_wd = []  
    start_idx = 0  # Keeps track of where the valid segment starts 

    # Loop through nonuniform distances (starting at the second point and ending on the point before last)
    for i in range(1, len(dist)):

        # Check if a large gap is detected in the gap between the last and current point in the segment (valid_segments[i - 1] is false, gap is detected so preceed)
        if not valid_segments[i - 1]:  

            # Checks for cases where we have consecutive large gaps or a large gap at the begining (avoids having empty tuples)
            if start_idx < i:

                # Store distance and data for ith valid segment as a tuple in the segments list
                segments.append((dist[start_idx:i], data[start_idx:i])) 
                segments_wd.append((dist[start_idx:i], water_depth[start_idx:i]))     

            # Start the indec for the new segment
            start_idx = i 

    # Append the last segment if valid 
    if start_idx < len(dist):
        segments.append((dist[start_idx:], data[start_idx:]))
        segments_wd.append((dist[start_idx:], water_depth[start_idx:]))

    # Initialize masked output array correctly
    data_int = np.ma.masked_all(shape=dist_int.shape, dtype=np.float64)
    water_depth_int = np.ma.masked_all(shape=dist_int.shape, dtype=np.float64)

    #--- Perform interpolation on each valid segment for data ---# 

    # Loop through the distance and data for each valid segments
    for dist_seg, data_seg in segments:

        # Check if there are at least two points for interpolation
        if len(dist_seg) > 1:  

            # Create an interpolation function with linear method without extrapolation
            f_interp = interp1d(dist_seg, data_seg, kind='linear', bounds_error=False, fill_value=np.nan)

            # Select the new interpolation points
            if dir == -1:
                dist_n = dist_int[(dist_int >= dist_seg[-1]) & (dist_int <= dist_seg[0])]   
            else: 
                dist_n = dist_int[(dist_int >= dist_seg[0]) & (dist_int <= dist_seg[-1])]  

            # Interpolate
            data_n = f_interp(dist_n)

            # Fill in interpolated values while masking NaNs
            mask = np.isnan(data_n)
            data_int[np.isin(dist_int, dist_n)] = np.ma.array(data_n, mask=mask)

    #--- Perform interpolation on each valid segment for water depth ---# 

    # Loop through the distance and data for each valid segments
    for dist_seg, wd_seg in segments_wd:

        # Check if there are at least two points for interpolation
        if len(dist_seg) > 1:  

            # Create an interpolation function with linear method without extrapolation
            f_interp = interp1d(dist_seg, wd_seg, kind='linear', bounds_error=False, fill_value=np.nan)

            # Select the new interpolation points
            if dir == -1:
                dist_n = dist_int[(dist_int >= dist_seg[-1]) & (dist_int <= dist_seg[0])]   
            else: 
                dist_n = dist_int[(dist_int >= dist_seg[0]) & (dist_int <= dist_seg[-1])]  

            # Interpolate
            wd_n = f_interp(dist_n)

            # Fill in interpolated values while masking NaNs
            mask = np.isnan(wd_n)
            water_depth_int[np.isin(dist_int, dist_n)] = np.ma.array(wd_n, mask=mask)


    ##################################################################
    ## STEP #2 - Split record into on-shelf, transiton, and off-shelf regions 
    ##################################################################

    # Fill masked values with interpolation (or NaN)
    water_depth_filled = water_depth_int.filled(np.nan)  # Convert to a normal array with NaNs

    # Interpolate missing values before applying smoothing
    valid = ~np.isnan(water_depth_filled)
    water_depth_filled[~valid] = np.interp(np.flatnonzero(~valid), np.flatnonzero(valid),water_depth_filled[valid])

    # Smooth the water depth record using a gaussian filter so that spikes and valleys are not identified in another section 
    water_depth_filtered = np.ma.array(gaussian_filter1d(water_depth_filled, sigma=5))

    # Convert back to a masked array, keeping the original mask
    water_depth_sm = np.ma.array(water_depth_filtered, mask=water_depth_int.mask)

    ##################################################################
    ## STEP #2a - On-Shelf 
    ##################################################################

    # Find the indices for on-shelf 
    idx_on = water_depth_sm >= on_lim

    # Grab distance, data, and water depth from binned data
    dist_on = dist_int[idx_on]
    data_on = data_int[idx_on]
    water_depth_on = water_depth_int[idx_on]

    ##################################################################
    ## STEP #2b - Off-Shelf 
    ##################################################################

    # Find the indices for off and on shelf 
    idx_off = water_depth_sm <= off_lim

    # Grab distance, data, and water depth from binned data
    dist_off = dist_int[idx_off]
    data_off = data_int[idx_off]
    water_depth_off = water_depth_int[idx_off]

    ##################################################################
    ## STEP #2c - Transition Region
    ##################################################################

    # Extend the limits of the transition region to include a bit of the 
    upper_lim = trans_lim[0]
    lower_lim = trans_lim[1] 

    # Find the indices for the transition region 
    idx_trans = (water_depth_sm >= lower_lim) & (water_depth_sm <= upper_lim)

    # Grab distance, data, and water depth from binned data
    dist_trans = dist_int[idx_trans]
    data_trans = data_int[idx_trans]
    water_depth_trans = water_depth_int[idx_trans]

    
    ##################################################################
    ## STEP #3 - Detrend data record 
    ##################################################################not
    data_on_dt    = detrend(data_on, dist_on, mean = 0)      
    data_trans_dt = detrend(data_trans, dist_trans, mean = 0) 
    data_off_dt   = detrend(data_off, dist_off, mean = 0)     
    data_full_dt  = detrend(data_int, dist_int, mean = 0)   


    ##################################################################
    ## STEP #4 - Compute autocorrelation and Decorrelation Scale function 
    ##################################################################

    # Set parameters
    lag_on, lag_trans, lag_off, lag_full = len(data_on_dt), len(data_trans_dt), len(data_off_dt), len(data_full_dt)
    
    # Compute autocorrelation function normalized by zeroth lag (for averaging) 
    autocorr_on, _, _, _, dist_scale_on, _ = compute_autocorr(data_on_dt, dist_on, lag_on, estimator, 0)
    autocorr_trans, _, _, _, dist_scale_trans, _ = compute_autocorr(data_trans_dt, dist_trans, lag_trans, estimator, 0)
    autocorr_off, _, _, _, dist_scale_off, _ = compute_autocorr(data_off_dt, dist_off, lag_off, estimator, 0)

    # Compute autocorrelation function normalized by the first lag (for transect decorrelation)
    autocorr_on_n, _, _, _, dist_scale_on_n, _ = compute_autocorr(data_on_dt, dist_on, lag_on, estimator, 1)
    autocorr_trans_n, _, _, _, dist_scale_trans_n, _ = compute_autocorr(data_trans_dt, dist_trans, lag_trans, estimator, 1)
    autocorr_off_n, _, _, _, dist_scale_off_n, _ = compute_autocorr(data_off_dt, dist_off, lag_off, estimator, 1)

    # Cmpute the unbiased autocovariance function (for spectra computation)
    _, _, autocov_on, _, _, _ = compute_autocorr(data_on_dt, dist_on, lag_on, 'unbiased', 0)
    _, _, autocov_trans, _, _, _ = compute_autocorr(data_trans_dt, dist_trans, lag_trans, 'unbiased', 0)
    _, _, autocov_off, _, _, _ = compute_autocorr(data_off_dt, dist_trans, lag_off, 'unbiased', 0)
    _, _, autocov_full, _, _, _ = compute_autocorr(data_full_dt, dist_int, lag_full, 'unbiased', 0)

    # Compute decorrelation time scale 
    L_on    = compute_decor_scale(autocorr_on,dist_scale_on,L,estimator,1)
    L_trans = compute_decor_scale(autocorr_trans,dist_scale_trans,L,estimator,1)
    L_off   = compute_decor_scale(autocorr_off,dist_scale_off,L,estimator,1)


    ##################################################################
    ## STEP 5 - Plot water depth, data, and autocorrelation function
    ##################################################################

    # Set plotting parameters 
    fontsize = 14

    # Plot water depth, data, and autocorrelation  
    if option_plot == 1:
        plot_depth_data_autocorr(dist, dist_on, dist_trans, dist_off, water_depth, water_depth_on, water_depth_trans, water_depth_off, data, data_on, data_trans, data_off, autocorr_on, autocorr_trans, autocorr_off, on_lim, off_lim, fontsize, dirOut)

    return autocorr_on, autocorr_trans, autocorr_off, autocorr_on_n, autocorr_trans_n, autocorr_off_n, autocov_on, autocov_trans, autocov_off, autocov_full, L_on, L_trans, L_off, dist_on, dist_trans, dist_off, dist_int, dist_scale_on_n, dist_scale_trans_n, dist_scale_off_n



#--- Segment time series ---# 
def segment_time_series(time, data, segment_years=1, overlap=0.5):
    """
    Split a time series into overlapping segments.

    Parameters
    ----------
    time : ndarray of datetime64
        Time vector (1D array of datetime objects).
    data : ndarray
        Data vector (1D array aligned with `time`).
    segment_years : float, optional
        Length of each segment in years (default is 1).
        Can be fractional (e.g., 0.5 = 6 months).
    overlap : float, optional
        Fraction of overlap between consecutive segments (0–1).
        For example, 0.5 means 50% overlap.

    Returns
    -------
    segments : list of tuples
        Each entry is (time_segment, data_segment), where:
        - time_segment : ndarray of datetimes for that segment
        - data_segment : ndarray of data values for that segment
    """

    # Import libraries
    from datetime import timedelta

    # Start and end times of the full time series
    start_time = time[0]
    end_time = time[-1]

    # Step size between the starts of consecutive segments (in years)
    # Example: 1-year window with 50% overlap -> step = 0.5 years
    step = segment_years * (1 - overlap)

    # Store the (time, data) pairs for each segment
    segments = []

    # Initialize the first segment start
    seg_start = start_time

    while True:
        # Define the end time for this segment
        seg_end = seg_start + timedelta(days=int(365*segment_years))

        # If the segment would extend beyond the available record, stop
        if seg_end > end_time:
            break

        # Create a mask to select time points within this segment
        mask = (time >= seg_start) & (time < seg_end)

        # Append the selected time and data as one segment
        segments.append((time[mask], data[mask]))

        # Move the start time forward by the step (handles overlap)
        seg_start += timedelta(days=int(365*step))

    return segments



#--- Bootstrapping estimate ---#  
def bootstrap_decorrelation_scale(data, x, window_length, dx, bias='unbiased', norm=0,
                                  stride=1, m=None, B=1000, lag=None, random_seed=None):
    """
    Bootstrap estimation of the decorrelation scale from a 1D data record.

    Parameters
    ----------
    data : array
        Full data record (1D array). Make sure to filter data beforehand 
        (e.g., remove annual and semi-annual cycles)! 
    x : array
        Time or spatial vector corresponding to `data`.
    window_length : int
        Length of each sliding window (number of points).
    dx : float
        Distance between consecutive data points (in time or space units).
    bias : str
        'biased', 'unbiased', or 'unbiased_tapered' for autocorrelation computation.
    norm : int
        0 = normalize by zero-lag variance, 1 = normalize by first lag.
    stride : int
        Increment to slide the window across the time series.
    m : int or None
        Number of windows to sample per bootstrap replicate.
        If None, defaults to all windows (J) with replacement.
    B : int
        Number of bootstrap replicates.
    lag : int or None
        Maximum lag for autocorrelation. If None, defaults to window_length.
    random_seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    mean_tau : float
        Mean decorrelation scale across bootstrap replicates.
    std_tau : float
        Standard deviation of decorrelation scales.
    stdm_tau : float
        Standard error of mean decorrelation scales across bootstrap replicates.
    ci95 : tuple (float, float)
        95% confidence interval from bootstrap percentiles.
    tau_boot: array
        Bootstrap decorrelation scale samples. 
    """

    # Import functions
    import numpy as np
    from lsf import detrend
    from autocorr import compute_decor_scale_optimize, compute_autocorr_optimize

    # Initialize random number generator for reproducibility (for uniformly drawing m windows from collection)
    rng = np.random.default_rng(random_seed)

    # Set the total length of time series
    N = len(data)  

    # Set the number of lags for the autocorrelation function 
    if lag is None:
        lag = window_length  # default maximum lag is the window length

    ################################################################
    # --- Step 1: Determine starting indices for sliding windows ---
    ################################################################

    # Find starting indicies for sliding windows
    start_indices = np.arange(0, N - window_length + 1, stride)

    # Set total number of windows
    J_total = len(start_indices)  

    ################################################################
    # --- Step 2: Compute positive-lag autocorrelations for each window ---
    ################################################################

    # Initialize autocorrelation function array 
    acfs = []

    # Loop through starting indicies for J windows 
    for s in start_indices:

        # Extract window of data
        window   = data[s:s + window_length]
        x_window = x[s:s + window_length]

        # Detrend window 
        window_dt = detrend(window, x_window, mean = 0)

        # Compute autocorrelation using optimized function
        rho_pos, _, _, _, x_ref_pos, _ = compute_autocorr_optimize(window_dt, x_window,
                                                                   lag, bias, norm)
        
        # Store autocorrelation and corresponding lags
        acfs.append((rho_pos, x_ref_pos))

    # Convert lists to arrays for easier indexing
    rho_array = np.array([r for r, _ in acfs])
    x_ref_array = np.array([xr for _, xr in acfs])

    ################################################################
    # --- Step 3: Set number of windows per bootstrap replicate ---
    ################################################################

    #######
    # Note
    # ----
    # When m = J_total (the number of windows draw equals the total number of windows), we will 
    # sample with replacement because if we sample without replacement, we could get the same 
    # decorrelation scale estimate with each bootstrapping interation. This approach should 
    # only be used when the windows are independent of each other. If they are dependent, this 
    # would lead to an underestimation of the uncertainty. 
    # 
    # When m < J_total (the number of windows draw is less than the total number of windows), we will 
    # sample without replacement. This will help when the windows of more strongly correlated 
    # with each other so that the uncertainty is on underestimated. 

    # Specify whether we will use sampling with or without replacement. 
    if m is None:
        m = J_total           # default: use all windows
        with_replacement = True
    else:
        if m == J_total: 
            with_replacement = True   # subsampling with replacement
        if m < J_total: 
            with_replacement = False  # subsampling without replacement
        if m > J_total:
            raise ValueError("m cannot exceed the total number of windows.")

    ################################################################
    # --- Step 4: Run bootstrap ---
    ################################################################

    # Initialize bootstrapping tau array
    tau_boot = np.empty(B, dtype=float) 

    # Loop through bootstrapping iterations 
    for b in range(B):

        # Sample window indices
        if with_replacement:
            idx = rng.integers(0, J_total, size=m)  # with replacement
        else:
            idx = rng.choice(J_total, size=m, replace=False)  # without replacement

        # Compute mean autocorrelation across sampled windows
        mean_rho = np.nanmean(rho_array[idx, :], axis=0)

        # Use x_ref from first sampled window (assumes all windows same spacing)
        mean_x_ref = x_ref_array[idx[0], :]

        # Compute decorrelation scale using optimized function
        tau_boot[b] = compute_decor_scale_optimize(mean_rho, mean_x_ref, dx, 'unbiased', norm)

    ################################################################
    # --- Step 5: Compute statistics from bootstrap replicates ---
    ################################################################

    # Mean decorrelation scale
    mean_tau = np.mean(tau_boot)  

    # Standard deviation 
    std_tau = np.std(tau_boot, ddof=1) 

    # Standard error 
    stdm_tau = std_tau / np.sqrt(tau_boot.size)

    # 95% confidence intervals
    ci_lower, ci_upper = np.percentile(tau_boot, [2.5, 97.5])  # 95% CI

    return mean_tau, std_tau, stdm_tau, (ci_lower, ci_upper), tau_boot

##--- Windowing decorrelation scale Analysis (Mask Aware)---# 
def windowed_decorrelation_scale(data, t, dt, scales=None, norm=0, overlap=0.0, return_windows=False, valid_frac=0.8, detrend_option=True):

    """
    Compute the decorrelation as a function of scale (Mahadevan et al. 2002 style)
    for a 1D time series, with option for overlapping windows and masking support. 

    Parameters
    ----------
    data : array_like
        Input data time series (1D). Can contain NaNs, which will be masked.
    t : array_like
        Input time vector. Can contain NaNs, which will be masked.
    dt : float
        Time increment of time series. 
    scales : list or array, optional
        List of window sizes (in samples). Defaults to powers of 2 up to length of data.
    norm : bool, optional
        The norm argument in the autocorrelation and decorrelations functions (normalization of the autocorrelation function). 
    overlap : float, optional
        Fractional overlap between adjacent windows (0 = no overlap, 0.5 = 50% overlap).
    return_windows : bool, optional
        If True, also returns dictionary of per-window decorrelation scales.
    valid_frac : float, optional
        Fraction of valid (non-NaN) data required in each window. Default is 0.8.
    detrend_option : bool, optional
        If True, detrend each window before computing autocorrelation. Default is True.

    Returns
    -------
    scales : ndarray
        Array of scales (window lengths).
    Lt_av : np.ma.MaskedArray
        Decorrelation scale at each scale (masked where not valid).
    Lt_stdm: np.ma.MaskedArray
        Decorrelation scale standard error of the mean at each scale (masked where not valid).
    window_decor_dict : dict, optional
        Per-scale decorrelation scale of individual windows.
    """

    # Import libraries 
    import numpy as np 
    from lsf import detrend
    from autocorr import compute_decor_scale_optimize, compute_autocorr_optimize

    #-----------------------------------#
    # STEP #1 - Compute Decorrelation Scale
    #-----------------------------------#

    # Set nans to masked values and set total length of the time series
    data = np.ma.masked_invalid(data)
    t = np.ma.masked_invalid(t)
    N = len(data)

    # Validate overlap input
    if not (0 <= overlap < 1):
        raise ValueError("overlap must be between 0 and 1 (fraction)")

    # If no scales are provided, use powers of 2 up to the length of the time series
    if scales is None:
        max_power = int(np.floor(np.log2(N)))
        scales = [2**k for k in range(3, max_power+1)] # e.g., [8,16,32,...]

    # Ensure scales is a NumPy array
    scales = np.array(scales)

    # Compute the duration of each of the scales
    duration = scales * dt

    # Initialize list to store average decorrelation per scale and dictionary to 
    # optionally store per-window decorrelation scales
    Lt_av = np.ma.masked_all(len(scales))
    Lt_stdm = np.ma.masked_all(len(scales))
    window_decor_dict = {}

    # Loop over scale
    for i, L in enumerate(scales):

        # Start indices for windows of length L
        step = max(1, int(L * (1 - overlap)))
        starts = np.arange(0, N - L + 1, step)

        # Skip if the window size is too large for the series
        if len(starts) == 0:
            continue 

        # Store decorrelation scale of each window
        window_autocorr = []
        window_lag      = []
        window_decor    = []

        # Loop through windows
        for s in starts:

            # Extract segment/window
            segment = data[s:s+L]
            t_seg   = t[s:s+L]

            # Require >= valid_frac fraction of valid data
            if (segment.count() >= valid_frac * L) and (t_seg.count() >= valid_frac * L):

                # Detrend window 
                if detrend_option:
                    segment_dt = detrend(segment,t_seg,mean=0)
                else:
                    segment_dt = segment

                # Compute autocorrelation function
                autocorr, _, _, _, lag, _ = compute_autocorr_optimize(segment_dt, t_seg, len(segment), 'biased', norm)

                # Compute the decorrelation scale
                seg_decor = compute_decor_scale_optimize(autocorr,lag,dt,'unbiased',norm)

                # Save the autocorrelation and decorrelation scales 
                window_autocorr.append(autocorr)
                window_lag.append(lag)
                window_decor.append(seg_decor)

        # Skip if no valid windows
        if len(window_decor) == 0:
            continue

        # Store per-window decorrelation scales
        window_decor_dict[L] = window_decor

        # Convert window_autocor from a list to an array
        window_autocorr = np.array(window_autocorr)
        window_lag = np.array(window_lag)

        # Average autocorrelation function across all windows for this scale 
        window_autocorr_av = np.nanmean(window_autocorr,axis=0)
        window_lag_av = np.nanmean(window_lag,axis=0)

        # Compute the standard deviation and the number of samples 
        window_autocorr_std = np.nanstd(window_autocorr, axis=0, ddof=1)
        window_autocorr_n   = np.count_nonzero(~np.isnan(window_autocorr), axis=0)

        # Compute the standard error of the mean (assuming that each data point is an independent observations)
        window_autocorr_stdm   = window_autocorr_std/np.sqrt(window_autocorr_n)

        # Skip if averaged autocorr is all NaN
        if np.all(np.isnan(window_autocorr_av)):
            continue

        # Compute average decorrelation scale and uncertainty
        try:

            # Compute average decorrelation scale
            window_decor_av = compute_decor_scale_optimize(
                window_autocorr_av, window_lag_av, dt, 'biased', norm
            )

            # Estimate the decorrelation scale's uncertainty by perturbing the autocorrelation function by 1 standard deviation
            window_decor_ustd = compute_decor_scale_optimize(window_autocorr_av + window_autocorr_stdm, 
                                                    window_lag_av ,dt,'unbiased',norm)
            window_decor_lstd = compute_decor_scale_optimize(window_autocorr_av - window_autocorr_stdm, 
                                                    window_lag_av ,dt,'unbiased',norm)

            # Compute the average standard error 
            std_upper = window_decor_ustd - window_decor_av
            std_lower = abs(window_decor_lstd - window_decor_av)
            window_decor_stdm  = np.mean([std_upper, std_lower])

        except ValueError:

            # catches the "array of sample points is empty" error
            continue

        # Save the decorrelation scale
        Lt_av[i] = window_decor_av 
        Lt_stdm[i] = window_decor_stdm

    # Return results, optionally including per-window decorrelation scales
    if return_windows:
        return scales, Lt_av, Lt_stdm, window_decor_dict
    else:
        return scales, Lt_av, Lt_stdm



##--- Windowing decorrelation scale Analysis (Mask Aware) with Boot Strap method---# 
def windowed_decorrelation_scale_boot(data, t, dt, scales=None, norm=False, overlap=0.0, return_windows=False, valid_frac=0.8):

    """
    Compute the decorrelation as a function of scale (Mahadevan et al. 2002 style)
    for a 1D time series using bootstrapping approach.  

    Parameters
    ----------
    data : array_like
        Input data time series (1D). Can contain NaNs, which will be masked.
    t : array_like
        Input time vector. Can contain NaNs, which will be masked.
    dt : float
        Time increment of time series. 
    scales : list or array, optional
        List of window sizes (in samples). Defaults to powers of 2 up to length of data.
    norm : bool, optional
        If True, the decorrelation scale is normalized by the duration of the record. 
    overlap : float, optional
        Fractional overlap between adjacent windows (0 = no overlap, 0.5 = 50% overlap).
    return_windows : bool, optional
        If True, also returns dictionary of per-window decorrelation scales.
    valid_frac : float, optional
        Fraction of valid (non-NaN) data required in each window. Default is 0.8.

    Returns
    -------
    scales : ndarray
        Array of scales (window lengths).
    Lt_av : np.ma.MaskedArray
        Decorrelation scale at each scale (masked where not valid).
    Lt_stdm: np.ma.MaskedArray
        Decorrelation scale standard error of the mean at each scale (masked where not valid).
    window_decor_dict : dict, optional
        Per-scale decorrelation scale of individual windows.
    """

    # Import libraries 
    import numpy as np 
    from lsf import detrend
    from autocorr import compute_decor_scale_optimize, compute_autocorr_optimize

    #-----------------------------------#
    # STEP #1 - Compute tau(L)
    #-----------------------------------#

    # Set nans to masked values and set total length of the time series
    data = np.ma.masked_invalid(data)
    t = np.ma.masked_invalid(t)
    N = len(data)

    # Validate overlap input
    if not (0 <= overlap < 1):
        raise ValueError("overlap must be between 0 and 1 (fraction)")

    # If no scales are provided, use powers of 2 up to the length of the time series
    if scales is None:
        max_power = int(np.floor(np.log2(N)))
        scales = [2**k for k in range(3, max_power+1)] # e.g., [8,16,32,...]

    # Ensure scales is a NumPy array
    scales = np.array(scales)

    # Compute the duration of each of the scales
    duration = scales * dt

    # Initialize list to store average decorrelation per scale and dictionary to 
    # optionally store per-window decorrelation scales
    Lt_av = np.ma.masked_all(len(scales))
    Lt_stdm = np.ma.masked_all(len(scales))
    window_decor_dict = {}

    # Loop over scale
    for i, L in enumerate(scales):

        # Start indices for windows of length L
        step = max(1, int(L * (1 - overlap)))
        starts = np.arange(0, N - L + 1, step)

        # Skip if the window size is too large for the series
        if len(starts) == 0:
            continue 

        # Store decorrelation scale of each window
        window_autocorr = []
        window_lag      = []
        window_decor    = []

        # Loop through windows
        for s in starts:

            # Extract segment/window
            segment = data[s:s+L]
            t_seg   = t[s:s+L]

            # Require >= valid_frac fraction of valid data
            if (segment.count() >= valid_frac * L) and (t_seg.count() >= valid_frac * L):

                # Detrend window 
                segment_dt = detrend(segment,t_seg,mean=0)

                # Compute autocorrelation function
                autocorr, _, _, _, lag, _ = compute_autocorr_optimize(segment_dt, t_seg, len(segment), 'biased', 0)

                # Compute the decorrelation scale
                seg_decor = compute_decor_scale_optimize(autocorr,lag,dt,'unbiased',0)

                # Save the autocorrelation and decorrelation scales 
                window_autocorr.append(autocorr)
                window_lag.append(lag)
                window_decor.append(seg_decor)

        # Skip if no valid windows
        if len(window_decor) == 0:
            continue

        # Normalize by duration and store per-window decorrelation scales
        if norm:
            window_decor = np.array(window_decor) / duration[i]
        window_decor_dict[L] = window_decor

        # Convert window_autocor from a list to an array
        window_autocorr = np.array(window_autocorr)
        window_lag = np.array(window_lag)

        # Average autocorrelation function across all windows for this scale 
        window_autocorr_av = np.nanmean(window_autocorr,axis=0)
        window_lag_av = np.nanmean(window_lag,axis=0)

        # Compute the standard deviation and the number of samples 
        window_autocorr_std = np.nanstd(window_autocorr, axis=0, ddof=1)
        window_autocorr_n   = np.count_nonzero(~np.isnan(window_autocorr), axis=0)

        # Compute the standard error of the mean (assuming that each data point is an independent observations)
        window_autocorr_stdm   = window_autocorr_std/np.sqrt(window_autocorr_n)

        # Skip if averaged autocorr is all NaN
        if np.all(np.isnan(window_autocorr_av)):
            continue

        # Compute average decorrelation scale and uncertainty
        try:

            # Compute average decorrelation scale
            window_decor_av = compute_decor_scale_optimize(
                window_autocorr_av, window_lag_av, dt, 'unbiased', 0
            )

            # Estimate the decorrelation scale's uncertainty by perturbing the autocorrelation function by 1 standard deviation
            window_decor_ustd = compute_decor_scale_optimize(window_autocorr_av + window_autocorr_stdm, 
                                                    window_lag_av ,dt,'unbiased',0)
            window_decor_lstd = compute_decor_scale_optimize(window_autocorr_av - window_autocorr_stdm, 
                                                    window_lag_av ,dt,'unbiased',0)

            # Compute the average standard error 
            std_upper = window_decor_ustd - window_decor_av
            std_lower = abs(window_decor_lstd - window_decor_av)
            window_decor_stdm  = np.mean([std_upper, std_lower])

        except ValueError:

            # catches the "array of sample points is empty" error
            continue

        # Normalize the decorrelation scale
        if norm:
            Lt_av[i]   = window_decor_av / duration[i]
            Lt_stdm[i] = window_decor_stdm / duration[i]
        else: 
            Lt_av[i] = window_decor_av 
            Lt_stdm[i] = window_decor_stdm

    # Return results, optionally including per-window decorrelation scales
    if return_windows:
        return scales, Lt_av, Lt_stdm, window_decor_dict
    else:
        return scales, Lt_av, Lt_stdm
    

#--- Computing decorrelation scale mean and uncertainty from an ensemble ---# 
def compute_decorrelation_stats(autocorr, lag, dx, compute_decor_scale_optimize, sample_axis=0):
    """
    Compute mean, standard deviation, and uncertainty estimates of the
    decorrelation scale from an ensemble of autocorrelation functions.

    Parameters
    ----------
    autocorr : ndarray
        Array of autocorrelation functions, typically shape (n_samples, n_lags).
        Each row corresponds to one realization of the autocorrelation function.
    lag : ndarray
        1D array of spatial or temporal lags corresponding to the lag dimension.
    dx : float
        Spatial or temporal resolution used in the decorrelation scale integration.
    compute_decor_scale_optimize : callable
        Function to compute the decorrelation scale. Must accept arguments:
        (autocorr, lag, dx, norm_type, zero_crossing_flag).
    sample_axis : int, default=0
        Axis corresponding to independent samples (over which to average).

    Returns
    -------
    L_mean : float
        Mean decorrelation scale computed from the ensemble-mean autocorrelation.
    L_stdm : float
        Estimated uncertainty (standard error) of the decorrelation scale.
    autocorr_m : ndarray
        Mean autocorrelation function across samples.
    autocorr_stdm : ndarray
        Standard error of the mean autocorrelation function across samples.
    """

    # Import libraries
    import numpy as np

    # --- Validate sample_axis ---
    if sample_axis < 0:
        sample_axis = autocorr.ndim + sample_axis
    if sample_axis >= autocorr.ndim:
        raise ValueError(f"Invalid sample_axis={sample_axis} for autocorr with shape {autocorr.shape}")

    # --- Compute statistics along sample dimension ---
    autocorr_m = np.nanmean(autocorr, axis=sample_axis)
    autocorr_std = np.nanstd(autocorr, axis=sample_axis, ddof=1)
    autocorr_n = np.count_nonzero(~np.isnan(autocorr), axis=sample_axis)
    autocorr_stdm = autocorr_std / np.sqrt(autocorr_n)

    # --- Compute decorrelation scale for mean autocorrelation ---
    L_mean = compute_decor_scale_optimize(autocorr_m, lag, dx, 'unbiased', 0)

    # --- Perturb by ±1 standard deviation of the mean ---
    L_upper = compute_decor_scale_optimize(autocorr_m + autocorr_stdm, lag, dx, 'unbiased', 0)
    L_lower = compute_decor_scale_optimize(autocorr_m - autocorr_stdm, lag, dx, 'unbiased', 0)

    # --- Compute uncertainty estimate ---
    std_upper = L_upper - L_mean
    std_lower = abs(L_lower - L_mean)
    L_stdm = np.mean([std_upper, std_lower])

    return L_mean, L_stdm, autocorr_m, autocorr_stdm
    


#--- Compute multiple estimates of the decorrelation scale ---# 
def compute_decorrelation_scales(rho, lags, max_lag_fit=None):

    """
    Compute multiple decorrelation scales from a positive lag autocorrelation function.

    Parameters
    ----------
    rho : array
        Positive lag autocorrelation function.
    lags : array
        Positive lag values (time or space) corresponding to rho.
    max_lag_fit : float, optional
        Maximum lag to use when fitting exponential model. 
        If None, use first quarter of lags.

    Returns
    -------
    scales : dict
        Dictionary with estimates for (all are one-sided unless labeled 'full'):
        - 'efolding'        : lag > 0 where rho = 1/e
        - 'efolding_full'   : 2 * efolding (for symmetric ACF)
        - 'zero_crossing'   : first positive lag where rho crosses 0
        - 'zero_crossing_full' : 2 * zero_crossing
        - 'hwhm'            : half-width at half-maximum (rho = 0.5)
        - 'fwhm'            : full-width at half-maximum = 2 * hwhm
        - 'exp_fit'         : L from rho ~ exp(-lag / L) (positive lags)
        - 'exp_fit_full'    : 2 * exp_fit (for symmetric exponential ACF)
    """

    # Import library
    import numpy as np
    from scipy.optimize import curve_fit

    # Dictionary to store results
    scales = {}

    # Ensure inputs are numpy arrays
    rho = np.asarray(rho)
    lags = np.asarray(lags)

    # ============================================================
    # 1. e-Folding scale (lag where rho = 1/e ≈ 0.3679)
    # ============================================================

    # Define target value
    target = 1/np.e

    # Find indices where rho drops below the target
    below = np.where(rho <= target)[0]

    # If a crossing exists and not at lag=0
    if len(below) > 0 and below[0] > 0:
        # Index of first point below the target
        i = below[0]
        # Linearly interpolate between the last point above and the first below
        L_e = np.interp(target, [rho[i-1], rho[i]], [lags[i-1], lags[i]])
    else:
        # If no crossing, return NaN
        L_e = np.nan

    # Store one and two-sided estimates
    scales["efolding"] = L_e
    scales["efolding_full"] = 2 * L_e if np.isfinite(L_e) else np.nan

    # ============================================================
    # 2. Zero-crossing scale (lag where rho first crosses 0)
    # ============================================================

    # Find indices where rho drops below or equal to 0
    below = np.where(rho <= 0)[0]

    # If a crossing exists
    if len(below) > 0 and below[0] > 0:
        # Index of first point below zero
        i = below[0]
        # Interpolate to find the exact crossing point
        zc = np.interp(0, [rho[i-1], rho[i]], [lags[i-1], lags[i]])
    else:
        # If no crossing, return NaN
        zc = np.nan

    # Store one and two-sided estimates
    scales["zero_crossing"] = zc
    scales["zero_crossing_full"] = 2 * zc if np.isfinite(zc) else np.nan

    # ============================================================
    # 3. Half-width at half-maximum (lag where rho = 0.5)
    # ============================================================

    # Find indices where rho drops below 0.5
    below = np.where(rho <= 0.5)[0]

    # If a crossing exists
    if len(below) > 0 and below[0] > 0:
        # Index of first point below 0.5
        i = below[0]
        # Interpolate to find the half-width
        hwhm = np.interp(0.5, [rho[i-1], rho[i]], [lags[i-1], lags[i]])
    else:
        # If no crossing, return NaN
        hwhm = np.nan

    # Store one and two-sided estimates
    scales["hwhm"] = hwhm
    scales["fwhm"] = 2 * hwhm if np.isfinite(hwhm) else np.nan

    # ============================================================
    # 4. Exponential fit (fit rho ~ exp(-lag/L) to small-lag ACF)
    # ============================================================

    # If no maximum lag for fitting is given, default to first quarter of lags
    if max_lag_fit is None:
        max_lag_fit = lags[len(lags)//4]

    # Create mask to restrict fitting region
    mask = (lags >= 0) & (lags <= max_lag_fit)

    # Define exponential model for curve fitting
    def exp_model(lag, L):
        return np.exp(-lag / L)

    try:
        # Fit exponential model to rho within mask region
        popt, _ = curve_fit(exp_model, lags[mask], rho[mask], p0=[lags[1]])
        # Extract fitted scale parameter
        L_fit = popt[0]
    except Exception:
        # If fitting fails, return NaN
        L_fit = np.nan

    # Store one and two-sided estimates
    scales["exp_fit"] = L_fit
    scales["exp_fit_full"] = 2 * L_fit if np.isfinite(L_fit) else np.nan

    return scales

