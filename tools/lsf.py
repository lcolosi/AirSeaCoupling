# Least Squares-Fit functions
## Luke Colosi | lcolosi@ucsd.edu | December 22nd, 2020

########### Detrend Function ###########
def detrend(data,x,mean = 0):

    """
    detrend(data,mean = 0)

        Function for removing a linear trend from a 1 dimensional data series using a unweighted least square fit. 

        Parameters
        ----------
        data : array
            Data record to detrend
        mean : boolean
            Specifies whether the mean is removed or not. If 0, the mean is removed. If 1, the mean is retained.  

        Returns
        -------
        data_detrend : array
            Detrended signal. 

        Libraries necessary to run function
        -----------------------------------
        import numpy as np
        from unweighted_least_square_fit import least_square_fit

    """

    # import libraries
    from lsf import unweighted_lsf

    # Fit a linear trend at grid point
    data_trend, x_data, x_data_sigma, L2_norm = unweighted_lsf(
        data, x, parameters = 0, freqs = [], sigma = None
    )

    # Remove linear trend
    if mean == 0: 
        data_detrend = data - data_trend 
    elif mean == 1:
        data_detrend = data - data_trend + x_data[0] 

    return data_detrend


########### Unweighted least squares fit Function ###########
def unweighted_lsf(data, x, parameters, freqs, sigma):
    
    """
    unweighted_lsf(data, x, parameters, freqs, sigma)
    
    Function for computing a unweighted least squares fit to 1D data for
    a sinusoidal signal.

    Parameters
    ----------
    data : array
        Data record as a masked array. This data may contain masked 
        values for grid points of missing data. 

    x : array
        x-coordinates positions of data. Cannot contain masked values. This is a crucial point
        because the coordinate steps with masked data point are removed to compute the least square fit, 
        but the fit is evaluated at the these steps after the parameters are computed.

    parameters : array
        Specifies the number of frequencies fitted. Values for
        this parameter may range from 0 to 4 which 0 corresponds to a 
        linear least squares fit.  

    freqs : array
        Specifies the frequencies being fit in an array:
        [w1, w2, ..., wn] where the number of frequencies must be
        the same as the number of parameters.

    sigma : float
        Uncertainty in each data measurement. This code currently
        only accepts a scalar value for this argument.

    Returns
    -------
    hfit : array
        Unweighted Least squares fit model.

    x_data : array
        Coefficients of the model.

    x_data_sigma : array
        Uncertainty in model coefficients (Standard Deviation).

    L2_norm : float 
        L2 norm (minimized quantity)

    Libraries necessary to run function
    -----------------------------------
    import numpy as np
    """

    # Import libraries 
    import numpy as np

    # Check if data is a masked array
    assert type(data) == np.ma.core.MaskedArray, "Data is not a masked array"

    # Remove masked data points 

    #--- Masked data points exist ---# 
    if np.size(data.mask) > 1:

        # Set mask
        ind = data.mask

        # Remove masked data points from time and data
        x_n = x[~ind]
        data_n = data[~ind]

    #--- No masked values exist ---#
    elif np.size(data.mask) == 1:

        # Set new variables
        x_n = x
        data_n = data
    
    # Perform Least squares fit based on the number of parameters (0 sinusoid fit, linear fit)
    if parameters == 0:

        # Create Kernel Matrix for least squares fit (linear fit)
        A = np.vstack([np.ones(len(data_n)), x_n]).T
        x_data = np.linalg.inv(A.T @ A) @ A.T @ data_n
        
        # Compute fit
        hfit = x_data[0] + x_data[1] * x
    
    elif parameters == 1:

        # Create Kernel Matrix for least squares fit (1 sinusoid fit)
        A = np.vstack([np.ones(len(data_n)), 
                       np.sin(freqs[0] * x_n), 
                       np.cos(freqs[0] * x_n)]).T
        x_data = np.linalg.inv(A.T @ A) @ A.T @ data_n
        
        # Compute fit
        hfit = x_data[0] + x_data[1] * np.sin(freqs[0] * x) + x_data[2] * np.cos(freqs[0] * x)
    
    elif parameters == 2:

        # Create Kernel Matrix for least squares fit (2 sinusoid fit)
        A = np.vstack([np.ones(len(data_n)), 
                       np.sin(freqs[0] * x_n), np.cos(freqs[0] * x_n),
                       np.sin(freqs[1] * x_n), np.cos(freqs[1] * x_n)]).T
        x_data = np.linalg.inv(A.T @ A) @ A.T @ data_n
        
        # Compute fit
        hfit = x_data[0] + x_data[1] * np.sin(freqs[0] * x) + x_data[2] * np.cos(freqs[0] * x) \
               + x_data[3] * np.sin(freqs[1] * x) + x_data[4] * np.cos(freqs[1] * x)
    
    elif parameters == 3:

        # Create Kernel Matrix for least squares fit (3 sinusoid fit)
        A = np.vstack([np.ones(len(data_n)),
                       np.sin(freqs[0] * x_n), np.cos(freqs[0] * x_n),
                       np.sin(freqs[1] * x_n), np.cos(freqs[1] * x_n),
                       np.sin(freqs[2] * x_n), np.cos(freqs[2] * x_n)]).T
        x_data = np.linalg.inv(A.T @ A) @ A.T @ data_n
        
        # Compute fit
        hfit = x_data[0] + x_data[1] * np.sin(freqs[0] * x) + x_data[2] * np.cos(freqs[0] * x) \
               + x_data[3] * np.sin(freqs[1] * x) + x_data[4] * np.cos(freqs[1] * x) \
               + x_data[5] * np.sin(freqs[2] * x) + x_data[6] * np.cos(freqs[2] * x)
    
    elif parameters == 4:

        # Create Kernel Matrix for least squares fit (4 sinusoid fit)
        A = np.vstack([np.ones(len(data_n)),
                       np.sin(freqs[0] * x_n), np.cos(freqs[0] * x_n),
                       np.sin(freqs[1] * x_n), np.cos(freqs[1] * x_n),
                       np.sin(freqs[2] * x_n), np.cos(freqs[2] * x_n),
                       np.sin(freqs[3] * x_n), np.cos(freqs[3] * x_n)]).T
        x_data = np.linalg.inv(A.T @ A) @ A.T @ data_n
        
        # Compute fit
        hfit = x_data[0] + x_data[1] * np.sin(freqs[0] * x) + x_data[2] * np.cos(freqs[0] * x) \
               + x_data[3] * np.sin(freqs[1] * x) + x_data[4] * np.cos(freqs[1] * x) \
               + x_data[5] * np.sin(freqs[2] * x) + x_data[6] * np.cos(freqs[2] * x) \
               + x_data[7] * np.sin(freqs[3] * x) + x_data[8] * np.cos(freqs[3] * x)
    
    # Compute covariance matrix
    if sigma is not None:
        C = (sigma**2) * np.linalg.inv(A.T @ A) @ np.linalg.inv(A.T @ A).T
    else:
        C = np.zeros((A.shape[1], A.shape[1]))
    
    # Compute the standard deviation of the coefficients
    x_data_sigma = np.sqrt(np.diagonal(C))
    
    # Compute the misfit between the model and data
    e = A @ x_data - data_n
    
    # Compute the L2 norm
    L2_norm = np.sqrt(np.sum(e**2))
    
    return hfit, x_data, x_data_sigma, L2_norm



########### Unweighted least squares sinusoidal parameters ###########
def compute_amp_phase(data, model, x_data, parameters):
    """
    Function for computing diagnostic quantities from a least squares sinusoidal fit:
      (1) Residual: Misfit of the model (data - model)
      (2) Root mean square error (RMSE)
      (3) Fraction of variance explained (FVE)
      (4) Amplitude of each sinusoidal frequency
      (5) Phase of each sinusoidal frequency

    Parameters
    ----------
    data : array_like
        1D data array (can contain NaNs).
    model : array_like
        Model fit to the data (same shape as `data`).
    x_data : array_like
        Coefficients from the sinusoidal model.
    parameters : int
        Number of frequencies in the fit (1 to 4).

    Returns
    -------
    res : ndarray
        Residual array (data - model), with NaNs removed.
    rms : float
        Root mean square error of the residuals.
    fve : float
        Fraction of variance explained by the model.
    amplitude : list of float
        Amplitudes for each sinusoidal component.
    phase : list of float
        Phases for each sinusoidal component (in radians).
    """

    # Import libraries 
    import numpy as np

    ##########################################################
    ## Initialize Variables 
    ##########################################################

    # Initialize amplitude list
    amplitude = []

    # Initialize phase list
    phase = []

    ##########################################################
    ## Remove NaN values from data 
    ##########################################################

    # Find indices where data is NaN
    idx_nans = np.isnan(data)

    # Extract non-NaN data values
    data_n = data[~idx_nans]

    # Extract corresponding non-NaN model values
    model_n = model[~idx_nans]

    ##########################################################
    ## Goodness of Fit Parameters 
    ##########################################################

    # Compute residual (misfit between data and model)
    res = data_n - model_n

    # Compute root mean square error of the residual
    rms = np.sqrt(np.mean(res**2))

    # Compute fraction of variance explained by the model
    fve = 1 - (np.sum(res**2) / np.sum((data_n - np.mean(data_n))**2))

    ##########################################################
    ## Amplitude and Phase Parameters 
    ##########################################################

    # Loop through each fitted frequency
    for i in range(parameters):
        # Extract sine coefficient for frequency i
        sin_coeff = x_data[2 * i + 1]

        # Extract cosine coefficient for frequency i
        cos_coeff = x_data[2 * i + 2]

        # Compute amplitude as sqrt(sin^2 + cos^2)
        amplitude.append(np.sqrt(sin_coeff**2 + cos_coeff**2))

        # Compute phase as atan2(sin, cos)
        phase.append(np.arctan2(sin_coeff, cos_coeff))

    # Return residual, RMS, FVE, amplitude, and phase
    return res, rms, fve, amplitude, phase



########### Unweighted least squares sinusoidal parameter uncertainty  ###########
def compute_amp_phase_unc(x_data, x_data_sigma, parameters):
    """
    Function for computing uncertainty in amplitude and phase estimates from a least squares
    sinusoidal fit using propagation of error.

    Parameters
    ----------
    x_data : array_like
        Coefficients from the sinusoidal model.
    x_data_sigma : array_like
        Uncertainties associated with each coefficient in `x_data`.
    parameters : int
        Number of frequencies in the fit (1 to 4).

    Returns
    -------
    sigma_amp : list of float
        Uncertainties in amplitude estimates for each sinusoidal component.
    sigma_phase : list of float
        Uncertainties in phase estimates for each sinusoidal component (in radians).
    """
    ##########################################################
    # Initialize Variables 
    ##########################################################

    # Initialize list for amplitude uncertainties
    sigma_amp = []

    # Initialize list for phase uncertainties
    sigma_phase = []

    ##########################################################
    ## Compute Uncertainty Estimates 
    ##########################################################

    # Loop through each fitted frequency
    for i in range(parameters):
        # Get indices for sine and cosine coefficients
        idx_start = 2 * i + 1
        idx_end = idx_start + 2

        # Compute uncertainty in amplitude and phase for this frequency
        sa, sp = amp_phase_unc(x_data[idx_start:idx_end], x_data_sigma[idx_start:idx_end])

        # Append results
        sigma_amp.append(sa)
        sigma_phase.append(sp)

    # Return uncertainties
    return sigma_amp, sigma_phase



########### Compute parameter uncertainty  ###########
def amp_phase_unc(x_data, x_data_sigma):

    """
    Function for computing the uncertainty in amplitude and phase given the model 
    coefficients and their uncertainties.

    Parameters
    ----------
    x_data : array_like
        Coefficients of model for one sinusoidal frequency: 
        [a_1, a_2] where x(t) = a_0 + a_1*sin(wt) + a_2*cos(wt)
    x_data_sigma : array_like
        Uncertainty in the coefficients of the model.

    Returns
    -------
    sigma_amp : float
        Uncertainty in amplitude estimate.
    sigma_phase : float
        Uncertainty in phase estimate.
    """

    # Import libraries 
    import numpy as np

    ##########################################################
    ## Initialize
    ##########################################################

    # Extract sine and cosine coefficients
    a_1 = x_data[0]
    a_2 = x_data[1]

    # Extract uncertainties in coefficients
    sig_1 = x_data_sigma[0]
    sig_2 = x_data_sigma[1]

    ##########################################################
    ## Compute Uncertainty
    ##########################################################

    # Compute uncertainty in amplitude using error propagation
    sigma_amp = (1 / np.sqrt(a_1**2 + a_2**2)) * np.sqrt((sig_1 * a_1)**2 + (sig_2 * a_2)**2)

    # Compute uncertainty in phase using error propagation
    sigma_phase = np.sqrt(
        (sig_1**2) * (-a_2 / (a_1**2 + a_2**2))**2 +
        (sig_2**2) * (a_1 / (a_1**2 + a_2**2))**2
    )

    return sigma_amp, sigma_phase

