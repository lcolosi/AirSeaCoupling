# General Statistical Analysis functions
## Luke Colosi | lcolosi@ucsd.edu 

#--- 1D Probability Density Function ---# 
def compute_pdf_1D(x,edges):

    """
    compute_pdf_1D(x,edges)

        Function for compute a 1D histogram and probability density function from the random variable x.  

        Parameters
        ----------
        x : array-like
            x random variable (must have no masked values)  
        edges : array-like
            Bin edges for the histogram.

        Returns
        -------
        counts : array-like
            The number of samples in each bin in the histogram. 
        pdf : array-like
            Probability density function defined on the X,Y-grid. Normalization is N*bin_area 
            where N is the total number of counts.
        bin_centers: array-like
            An array of the center of each bin. 

        Libraries necessary to run function
        -----------------------------------
        import numpy as np 

    """

    # import library
    import numpy as np

    # Compute counts for each bin 
    counts, _ = np.histogram(x, edges, density=False)
    #pdf = np.histogram(x, edges, density=True) # This should give us the same result as below

    # Compute bin width  
    bin_width = np.mean(np.diff(edges))

    # Normalize counts by the total number of counts and bin area 
    pdf = counts / (np.sum(counts) * bin_width)

    # Calculate bin centers from edges
    bin_centers = np.array([(edges[i] + edges[i+1]) / 2 for i in range(len(edges)-1)])

    return counts, pdf, bin_centers

#--- 2D Probability Density Function ---# 
def compute_pdf_2D(x,y,X,Y):

    """
    compute_pdf_2D(x,y,X,Y)

        Function for compute a 2D probability density function from two random variables x and y.  

        Parameters
        ----------
        x : first random variable in a numpy array (must have no masked values)  
        y : second random variable in a numpy array (must have no masked values) 
        X : Uniform meshgird for the x random variable 
        Y : Uniform meshgrid for the y random variable 

        Returns
        -------
        counts : The number of samples in each bin in the histogram.
        pdf : Probability density function defined on the X,Y-grid. Normalization is N*bin_area 
              where N is the total number of counts
        x_edges : Bin center along the x-coordinate axis (for plotting)
        y_edges : Bin center along the y-coordinate axis (for plotting)

        Libraries necessary to run function
        -----------------------------------
        import numpy as np 

    """

    # import library
    import numpy as np
    
    # Obtain bin edges from x and y meshgrid
    x_edges = np.unique(X[0, :])
    y_edges = np.unique(Y[:, 0])

    # Compute counts for each bin 
    counts, _, _ = np.histogram2d(x, y, bins=[x_edges, y_edges])

    # Compute bin width and bin area 
    x_bin_width = np.unique(np.diff(x_edges))[0]
    y_bin_width = np.unique(np.diff(y_edges))[0]
    bin_area = x_bin_width * y_bin_width

    # Normalize counts by the total number of counts and bin area 
    pdf = counts / (np.sum(counts) * bin_area)

    # Compute bin centers
    x_center = [(x_edges[i] + x_edges[i+1]) / 2 for i in range(len(x_edges)-1)]
    y_center = [(y_edges[i] + y_edges[i+1]) / 2 for i in range(len(y_edges)-1)]

    return counts, pdf, x_center, y_center

#--- Principle Axis ---# 
def principal_axis(x, y):

    """
    principal_axis(x, y)

        Function to compute the angle (theta) of the principal axis such that if
        the data x, y is rotated by theta, then the rotated data xhat, yhat has
        a covariance that vanishes.

        Parameters
        ----------
        x : array-like
            x data. Can contain NaNs.
        y : array-like
            y data. Can contain NaNs.

        Returns
        -------
        theta : float
            Angle of principal axis (CCW, reference east, going towards; units: degrees ranging from 0 to 360)
        C : ndarray
            Covariance matrix of non-rotated data.
        Cr : ndarray
            Covariance matrix of rotated data.
        xhat : ndarray
            Rotated x data.
        yhat : ndarray
            Rotated y data.
    """

    # import library
    import numpy as np

    # Remove NaNs from x and y
    x = np.array(x)
    y = np.array(y)
    valid_indices = ~np.isnan(x) & ~np.isnan(y)
    x = x[valid_indices]
    y = y[valid_indices]

    # Compute the covariance matrix of x and y
    C = np.cov(x, y)

    # Compute principal axes (mod operator for range to be from 0 to 360)
    theta = np.mod(0.5 * np.degrees(np.arctan2(2 * C[0, 1], C[0, 0] - C[1, 1])), 360)

    # Set CCW rotation matrix
    M = np.array([[np.cos(np.radians(theta)), np.sin(np.radians(theta))],
                  [-np.sin(np.radians(theta)), np.cos(np.radians(theta))]])

    # Rotate data
    rotated_data = np.dot(np.column_stack((x, y)), M.T)
    xhat = rotated_data[:, 0]
    yhat = rotated_data[:, 1]

    # Compute the covariance matrix of rotated data
    Cr = np.cov(xhat, yhat)

    return theta, C, Cr, xhat, yhat



#--- Compute Variance Ellipse ---# 
def compute_variance_ellipse(u, v, lat=None, use_weights=True):
    """
    Compute the spatial variance ellipse and rotated velocity components
    from 2D u,v velocity fields in a subregion.

    Parameters
    ----------
    u, v : 2D arrays
        Velocity components within the spatial subregion (same shape).
        Units: m/s (or consistent velocity units).
    lat : 2D array, optional
        Latitude field (degrees) for area weighting via cos(latitude).
        Only used if `use_weights=True`.
    use_weights : bool, default=True
        If True, apply area weighting proportional to cos(latitude).

    Returns
    -------
    theta : float
        Orientation of the major axis **in degrees clockwise from North**.
        (0° = North, 90° = East, 180° = South, 270° = West)
    a : float
        Standard deviation (sqrt of variance) along the semi-major axis.
    b : float
        Standard deviation (sqrt of variance) along the semi-minor axis.
    u_rot, v_rot : 2D arrays
        Velocity anomalies rotated into the major/minor coordinate system:
        - u_rot → along-major-axis component
        - v_rot → along-minor-axis component

    Notes
    -----
    - NaNs are ignored when computing covariance and means.
    - Area weighting ensures each grid cell contributes proportionally
      to its surface area on Earth (∝ cos(latitude)).
    - Orientation follows oceanographic convention (bearing from North,
      increasing clockwise).
    - Rotation follows the classical variance ellipse convention
      (Davis/Rudnick/Gille, SIO 221B Lecture 4):
         θ = ½ * atan2(2<uv>, <u²> − <v²>)
    """

    # Import Library 
    import numpy as np

    # -------------------------------------------------------------
    # 1. Flatten inputs and remove NaNs
    # -------------------------------------------------------------
    # We treat the 2D subregion as a collection of N samples.
    # Flattening simplifies covariance computations as 1D inner products.
    uu = np.ravel(u)
    vv = np.ravel(v)

    # Boolean mask of valid (non-NaN) points
    mask = np.isfinite(uu) & np.isfinite(vv)

    # Extract valid values
    uu = uu[mask]
    vv = vv[mask]

    # -------------------------------------------------------------
    # 2. Define weights (area weighting using cos(latitude))
    # -------------------------------------------------------------
    if lat is not None and use_weights:
        # Use latitude at each grid cell to compute weight ∝ cos(lat)
        ww = np.ravel(np.cos(np.deg2rad(lat)))[mask]
    else:
        # Default to uniform weighting (equal area assumption)
        ww = np.ones(mask.sum())

    # Normalize total weight
    W = ww.sum()

    # -------------------------------------------------------------
    # 3. Compute weighted means and anomalies
    # -------------------------------------------------------------
    # Weighted mean removes any large-scale offset in u,v fields.
    u_mean = np.sum(ww * uu) / W
    v_mean = np.sum(ww * vv) / W

    # Velocity anomalies relative to spatial mean
    u_prime = uu - u_mean
    v_prime = vv - v_mean

    # -------------------------------------------------------------
    # 4. Compute weighted covariance components
    # -------------------------------------------------------------
    # These are spatial (not temporal) covariances:
    # <u'²>, <v'²>, and <u'v'>
    Cuu = np.sum(ww * u_prime * u_prime) / W
    Cvv = np.sum(ww * v_prime * v_prime) / W
    Cuv = np.sum(ww * u_prime * v_prime) / W

    # Construct covariance matrix
    C = np.array([[Cuu, Cuv],
                  [Cuv, Cvv]])

    # -------------------------------------------------------------
    # 5. Eigen-decomposition → principal axes of the covariance matrix
    # -------------------------------------------------------------
    # Eigenvalues correspond to variances along the ellipse's axes.
    # Eigenvectors give the directions of those axes.
    eigvals, eigvecs = np.linalg.eigh(C)        # sorted ascending by default
    lam1, lam2 = eigvals[1], eigvals[0]         # major, minor variances

    # Semi-major (a) and semi-minor (b) standard deviations
    a = np.sqrt(lam1)
    b = np.sqrt(lam2)

    # Major-axis unit vector (east,north components)
    e_major = eigvecs[:, 1]

    # Orientation angle of major axis (radians, CCW from East, x-axis)
    theta = np.arctan2(e_major[1], e_major[0])

    # -------------------------------------------------------------
    # Note on eigen-decomposition vs. analytic solution
    # -------------------------------------------------------------
    # The eigen-decomposition of the 2×2 covariance matrix C is
    # mathematically equivalent to the analytic variance-ellipse
    # solution derived in Davis, Rudnick & Gille (SIOC 221B Lecture 4):
    #
    #   tan(2θ) = 2Cuv / (Cuu − Cvv)
    #   a²,b² = ½[(Cuu + Cvv) ± √((Cuu − Cvv)² + 4Cuv²)]
    #
    # In both approaches:
    #   • The eigenvectors give the principal-axis directions (θ)
    #   • The eigenvalues give the variances along those axes (a²,b²)
    #
    # Eigen-decomposition is used here because it is numerically
    # stable, automatically provides orthogonal unit vectors, and
    # generalizes to higher-dimensional covariance matrices.

    # -------------------------------------------------------------
    # 6. Convert to oceanographic convention (clockwise from North)
    # -------------------------------------------------------------
    # Convert to clockwise reference north
    theta_north = (np.pi/2 - theta) % (2 * np.pi)  # Units: radians

    # Convert to degrees
    theta_north_deg = np.degrees(theta_north)      

    # -------------------------------------------------------------
    # 6. Rotate velocity anomalies into the major/minor coordinate system
    # -------------------------------------------------------------
    # To express the field in the rotated coordinate frame:
    # Perform a clockwise rotation by θ so that the x'-axis
    # aligns with the major axis of the variance ellipse.
    cos_t, sin_t = np.cos(theta_north), np.sin(theta_north)

    # Recreate 2D anomaly fields (NaNs restored in original shape)
    u_prime_2d = np.full_like(u, np.nan)
    v_prime_2d = np.full_like(v, np.nan)
    u_prime_2d.flat[mask] = u_prime
    v_prime_2d.flat[mask] = v_prime

    # Apply clockwise rotation by θ:
    #   [u_rot]   [ cosθ   sinθ][u']
    #   [v_rot] = [-sinθ   cosθ][v']
    u_rot =  u_prime_2d * cos_t + v_prime_2d * sin_t   # along-major-axis component
    v_rot = -u_prime_2d * sin_t + v_prime_2d * cos_t   # along-minor-axis component

    # -------------------------------------------------------------
    # 7. Return results
    # -------------------------------------------------------------
    return theta_north_deg, a, b, u_rot, v_rot

#--- Bin Averaging Function ---# 
def masked_bin_average(data, bin_index, unique_bins):
    
    """
    Compute masked bin-averaged values from an array using a set of bin indices.

    Parameters
    ----------
    data : np.ma.MaskedArray (or array-like)
        Input data array containing values to be averaged. May include masked values.
    bin_index : array-like
        Array of the same shape as `data` indicating the bin assignment for each element.
    unique_bins : array-like
        Sorted unique bin labels over which the averaging is performed.

    Returns
    -------
    binned_avg : np.ma.MaskedArray
        A masked array of shape (len(unique_bins),) containing the mean value
        for each bin. Bins with no unmasked values are masked.
    """

    # import library
    import numpy as np

    # Define lists for mask and values
    vals = []
    mask = []

    # Loop through bins
    for b in unique_bins:

        # Grab data for ith bin
        d = data[bin_index == b]

        # Check if the bin is empty
        if np.ma.is_masked(d) and d.mask.all():

            # Fill will a dummy value (0.0) and set mask to true
            vals.append(0.0)    
            mask.append(True)
        else:

            # Fill with mean (ignoring masked values) and set mask to false
            vals.append(np.ma.mean(d))
            mask.append(False)

    # Return the averaged values with mask        
    return np.ma.array(vals, mask=mask)
