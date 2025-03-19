# Real-valued Autocorrelation Analysis functions
## Luke Colosi | lcolosi@ucsd.edu 

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
    R         = np.zeros(lag)   # Autocovariance
    rho       = np.zeros(lag)   # Autocorrelation 
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
    # The normalization is the same for the biased, unbiased, and unbiased estimates when normalizing with the zeroth lag
    #  because the factors in front of the inner product are equivalent: 
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






#--- Decorrelation Scale Analysis ---%
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
