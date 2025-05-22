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
    bin_width = np.unique(np.diff(edges))

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