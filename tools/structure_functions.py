# Structure Function Analysis
## Luke Colosi | lcolosi@ucsd.edu 

#--- Compute nth order Structure Function ---#
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

#--- Define models for fitting structure functions ---#
def _exp_sf_model(r, sigma2, L, offset):

    # Import libraries
    import numpy as np

    """
    Model for 2nd-order structure function:

    D2(r) = 2*sigma^2 * (1 - exp(-r/L)) + offset
    """

    # Ensure r is an array
    r = np.asarray(r, dtype=float)

    # Return exponential model values
    return 2.0 * sigma2 * (1.0 - np.exp(-r / L)) + offset

#--- Compute decorrelation scale from the structure function ---#
def compute_decor_scale_struct(r, S2, method="lsq", var0=None, fit_range=None, normalize=True, p0=None, bounds=None):

    """
    Function for estimating a decorrelation scale from a 1D 2nd-order structure function. This follows the todd et al. (2013) approach where
    an exponential model is fit to the structure function: 
        D2(r) = 2*sigma^2*(1 - exp(-r/L)) + offset
    and the e-folding scale L is returned.

    Two approaches implemented:

    (1) method="lsq"
        Fit an exponential model directly to the structure function:
            D2(r) = 2*sigma^2*(1 - exp(-r/L)) + offset
        and return the e-folding scale L.

    (2) method="lee2011"
        Convert structure function to autocovariance via:
            C(r) = C(0) - D2(r)/2
        Optionally convert to correlation (recommended) via:
            rho(r) = C(r)/C(0)
        Then fit a cosine–exponential model:
            rho(r) = offset + amp*exp(-r/L)*cos(2*pi*r/lam)
        returning L (decorrelation envelope) and lam (oscillation wavelength).

    Parameters
    ----------
    r : 1D ndarray
        Separation distances (time lags or spatial separations) corresponding to S2.
        Units of r set units of returned L.
    S2 : 1D ndarray
        Second-order structure function values D2(r).
    method : {"todd2013", "lee2011"}
        Which approach to use.
    var0 : float, optional
        Variance C(0). Required for method="lee2011" to convert S2 -> C(r) (and rho).
        If normalize=False, you still need var0 to compute C(r) from S2.
    fit_range : tuple (r_min, r_max), optional
        Range of r to include in the fit (in same units as r).
        Useful to exclude very small-lag noise and very large-lag saturation.
    normalize : bool
        For "lee2011": if True, fit correlation rho(r); if False, fit covariance C(r).
    p0 : tuple, optional
        Initial guess parameters for the chosen model.
        todd2013: (sigma2, L, offset)
        lee2011 : (L, lam, amp, offset)
    bounds : 2-tuple, optional
        Bounds for scipy.optimize.curve_fit: (lower_bounds, upper_bounds)

    Returns
    -------
    out : dict
        Fit parameters and metadata. Always includes:
            out["method"], out["L"], out["r_used"], out["S2_used"]
        Additionally:
            Todd: out["sigma2"], out["offset"]
            Lee:  out["lam"], out["amp"], out["offset"], out["fit_target"]
    """

    # Import libraries
    import numpy as np
    from scipy.optimize import curve_fit

    # Ensure inputs are arrays
    r = np.asarray(r, dtype=float)
    S2 = np.asarray(S2, dtype=float)

    # Keep only finite points and separations > 0 (we don't fit at r=0)
    m = np.isfinite(r) & np.isfinite(S2) & (r > 0)
    r0 = r[m]
    S20 = S2[m]

    # Apply an optional fit window
    if fit_range is not None:
        rmin, rmax = fit_range
        mw = (r0 >= rmin) & (r0 <= rmax)
        r0 = r0[mw]
        S20 = S20[mw]

    if r0.size < 5:
        raise ValueError(
            "Not enough valid points to fit. Increase max lag or widen fit_range."
        )
    
    # Define output dictionary with metadata
    out = {
        "method": method.lower(),
        "r_used": r0,
        "S2_used": S20,
    }
    method = method.lower()

    # =========================================================
    # (2) Todd et al. (2013) Approach: exponential SF fit to D2(r)
    # =========================================================
    if method == "todd2013":

        # -------------------------------------------------
        # Obtain initial guesses and bounds for model parameters 
        # -------------------------------------------------

        # Initial guess for offset: the nugget at r -> 0
        k = min(5, S20.size)
        offset_guess = max(float(np.nanmean(S20[:k])), 0.0)

        # Initial guess for sigma^2: from the large-lag plateau where S2(r -> inf) ~ 2*sigma^2 (+ offset)
        n_tail = max(3, S20.size // 5)                                            # Obtain number of the last 20% of points (ensure there are at least 3 points)
        sigma2_guess = max(np.nanmean(S20[-n_tail:] - offset_guess) / 2.0, 1e-12) # Estimate sigma^2 from tail (sigma^2 = S2(large r)/2)

        # Initial guess for decorrelation scale: The lag where S2 ~ 0.63 * (2*sigma^2)
        target = 0.632 * (2.0 * sigma2_guess)
        idx = np.nanargmin(np.abs(S20 - target))
        L_guess = r0[idx]

        # Set initial parameter guess tuple if not provided
        if p0 is None:
            p0 = (sigma2_guess, L_guess, offset_guess)

        # Set parameter bounds if not provided
        if bounds is None:

            # Constraints
            # -----------
            # (1) sigma2 >= 0 : Positive definite variance
            # (2) L > 0       : Positive definite decorrelation scale (choose a minimal positive L bound from spacing if possible)
            # (3) offset -> unconstrained
            if r0.size > 2:
                dr = np.min(np.diff(np.unique(r0)))
                L_lb = max(dr, 1e-12)
            else:
                L_lb = 1e-12
            bounds = ((0.0, L_lb, -np.inf), (np.inf, np.inf, np.inf))

        # -------------------------------------------------
        # Fit the model to the structure function data
        # -------------------------------------------------
        popt, _ = curve_fit(
            _exp_sf_model,
            r0,
            S20,
            p0=p0,
            bounds=bounds,
        )
        sigma2_hat, L_hat, offset_hat = popt
        
        # Update output dictionary with fit results
        out.update(
            {
                "sigma2": float(sigma2_hat),
                "L": float(L_hat),
                "offset": float(offset_hat),
            }
        )
        return out

    # =========================================================
    # (1) Lee et al. (2011) Approach: SF -> C(r) -> cosine-exp fit
    # =========================================================
    elif method == "lee2011":

        # Ensure variance is provided
        if var0 is None:
            raise ValueError(
                "For method='lee2011', you must provide var0 = variance (C(0))."
            )

        # -----------------------------------------------------------------------------
        # Converting 2nd-order structure function to autocovariance / autocorrelation
        # -----------------------------------------------------------------------------
        # For a (wide-sense) stationary process x with mean removed, the 2nd-order
        # structure function is:
        #
        #     D2(r) = < [x(s+r) - x(s)]^2 >
        #
        # Expanding the square and using stationarity gives:
        #
        #     D2(r) = <x^2> + <x^2> - 2<x(s+r)x(s)>
        #           = 2*C(0) - 2*C(r)
        #
        # where C(r) is the autocovariance and C(0) = Var(x).
        # Therefore:
        #
        #     C(r) = C(0) - D2(r)/2
        #
        # If you want the autocorrelation (dimensionless), divide by C(0):
        #
        #     rho(r) = C(r)/C(0) = 1 - D2(r) / (2*C(0))
        # -----------------------------------------------------------------------------

        # Convert structure function to autocovariance
        C = var0 - 0.5 * S20

        # Optionally convert to correlation for dimensionless fit
        if normalize:
            y = C / var0
            fit_target = "autocorrelation"
        else:
            y = C
            fit_target = "autocovariance"

        # -------------------------------------------------
        # Obtain initial guesses and bounds for model parameters 
        # -------------------------------------------------
        L_guess = np.median(r0)

        # Use a conservative wavelength guess based on sampling of r
        if r0.size > 2:
            dr = np.median(np.diff(np.unique(r0)))
        else:
            dr = r0[1] - r0[0]
        lam_guess = max(4.0 * dr, 2.0 * dr)

        # Amplitude: ~1 for correlation, ~var0 for covariance
        amp_guess = 1.0 if normalize else float(var0)

        # Offset: start with zero
        offset_guess = 0.0

        # Set initial parameter guess tuple if not provided
        if p0 is None:
            p0 = (L_guess, lam_guess, amp_guess, offset_guess)

        # Set parameter bounds if not provided
        if bounds is None:

            # Constraints
            # -----------
            # (1) L > 0       : Positive definite decorrelation scale (choose a minimal positive L bound from spacing if possible)
            # (2) lam > 0     : Positive definite oscillation wavelength
            # (3) amp/offset -> unconstrained
            bounds = ((1e-12, 1e-12, -np.inf, -np.inf), (np.inf, np.inf, np.inf, np.inf))

        # -------------------------------------------------
        # Fit the model to the structure function data
        # -------------------------------------------------
        popt, _ = curve_fit(
            _cosexp_corr_model,
            r0,
            y,
            p0=p0,
            bounds=bounds,
            maxfev=40000,
        )
        L_hat, lam_hat, amp_hat, offset_hat = popt

        # Update output dictionary with fit results
        out.update(
            {
                "fit_target": fit_target,
                "L": float(L_hat),
                "lam": float(lam_hat),
                "amp": float(amp_hat),
                "offset": float(offset_hat),
            }
        )
        return out

    else:
        raise ValueError("Unknown method. Use method='todd2013' or method='lee2011'.")
