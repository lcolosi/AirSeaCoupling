# Structure Function Analysis
## Luke Colosi | lcolosi@ucsd.edu 

def compute_structure_function(field, max_lag, orders=[1, 2, 3, 4]):
    """
    Compute nth-order structure functions using vectorized NumPy operations.
    Supports both 1D (single time series or profile) and 2D (multiple series/profiles) input.
    
    Parameters
    ----------
        field : 1D ndarray 
            Input data, where structure functions are computed.
        max_lag : int
            Maximum lag (spatial or temporal separation).
        orders : list of int
            List of orders (moments) to compute.
        
    Returns
    -------
        S : dictionary 
            Structure functions. Keys are order integers, values are arrays of shape (max_lag,) or (n_series, max_lag)
        lag : 1D ndarray
    """
    
    # Import libraries
    import numpy as np
    
    # Ensure the field is a NumPy array
    field = np.asarray(field)

    # Initialize dictionary for results
    S = {}

    # Set the lag vector
    lag = np.array([lag for lag in range(1, max_lag + 1)])

    # Compute structure functions by lag
    for order in orders:
        
        # For each lag, compute absolute difference to the power of the order
        diffs = [np.abs(field[lag:] - field[:-lag]) ** order for lag in range(1, max_lag + 1)]

        # For each lag, take the mean of the computed differences
        S[order] = np.array([np.mean(d) for d in diffs])

    # Return the dictionary of structure function arrays
    return S, lag