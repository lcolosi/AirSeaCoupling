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
