# Common Oceanographic Analysis functions
## Luke Colosi | lcolosi@ucsd.edu 

#--- Computing Mixed Layer ---#
def compute_mld(depth, temp=None, density=None, method='threshold', variable='density',
                dT=0.2, dSigma=0.03, gradT=0.025, gradSigma=0.0005,
                zref=10, phi=100, g=9.81):
    """
    Compute mixed layer depth (MLD) using threshold, gradient, or potential energy anomaly (PEA) criteria. Supports
    masked arrays for temperature and density.

    Parameters
    ----------
    depth : array_like
        1D array of depths (m, positive downward).
    temp : array_like
        1D array of potential temperature (°C).
    density : array_like
        1D array of potential density anomaly σθ (kg/m^3).
    method : {'threshold', 'gradient', 'potential_energy'}, optional
        Criterion to use for MLD definition.
    variable : {'density', 'temperature'}, optional
        Use density (σθ) or temperature to define MLD.
    dT : float, optional
        Temperature threshold (°C) for threshold method (default 0.2).
    dSigma : float, optional
        Density threshold (kg/m³) for threshold method (default 0.03).
    gradT : float, optional
        Temperature gradient threshold (°C/m) for gradient method (default 0.025).
    gradSigma : float, optional
        Density gradient threshold (kg/m⁴) for gradient method (default 0.0005).
    zref : float, optional
        Reference depth for threshold method (default 10 m).
    phi : float, optional
        Potential energy anomaly threshold (default 100 J/m^2).
    g : float, optional
        Gravitational acceleration (default 9.81 m/s²).

    Returns
    -------
    mld : float
        Mixed layer depth estimate (m). Returns np.nan if no criterion is met.
    """

    # Import libraries
    import numpy as np

    # Convert inputs to numpy arrays for safe indexing
    depth = np.asarray(depth)
    if temp is not None: temp = np.ma.masked_invalid(temp)
    if density is not None: density = np.ma.masked_invalid(density)

    # Ensure depth increases with index (surface first, deep last)
    if depth[0] > depth[-1]:
        depth = depth[::-1]
        if temp is not None: temp = temp[::-1]
        if density is not None: density = density[::-1]

    # ------------------------------
    # Helper Function: compress valid values
    # ------------------------------
    def valid_profile(arr):

        """
        Removes masked values from arr variable and depth profile leaving only valid (depth_valid, arr_valid) pairs. 
        """

        # Define mask for non-masked data-depth pairs (set mask to all False if arr is not a masked array)
        mask = ~arr.mask if np.ma.isMaskedArray(arr) else np.ones_like(arr, dtype=bool)

        # Ensure mask only keeps finite values (no NaN or +/- inf)
        mask &= np.isfinite(arr)

        # Handle case when no valid points remain (all masked or NaNs)
        if not np.any(mask):
            return None, None
        
        return depth[mask], arr[mask]

    # ------------------------------
    # Threshold method
    # ------------------------------
    if method == 'threshold':

        if variable == 'temperature':

            # Check if temperature variable is not None
            if temp is None:
                raise ValueError("Temperature array required for temperature-based MLD")
            
            # Remove masked values
            depth_valid, temp_valid = valid_profile(temp)
            if depth_valid is None:
                return np.nan
            
            # Find the index of the reference depth (default = 10 m)
            iref = np.argmin(np.abs(depth_valid - zref))
            
            # Compare each temp value with the reference temp
            ref_val = temp_valid[iref]
            diff = np.ma.abs(temp_valid - ref_val)

            # First depth where difference exceeds threshold dT
            idx = np.where(diff >= dT)[0]

        elif variable == 'density':

            # Check if density variable is not None
            if density is None:
                raise ValueError("Density array required for density-based MLD")
            
            # Remove masked values
            depth_valid, dens_valid = valid_profile(density)
            if depth_valid is None:
                return np.nan
            
            # Find the index of the reference depth (default = 10 m)
            iref = np.argmin(np.abs(depth_valid - zref))

            # Compare each density value with the reference density
            ref_val = dens_valid[iref]
            diff = dens_valid - ref_val

            # First depth where difference exceeds threshold dSigma
            idx = np.where(diff >= dSigma)[0]

        else:
            raise ValueError("variable must be 'temperature' or 'density'")

    # ------------------------------
    # Gradient method
    # ------------------------------
    elif method == 'gradient':
        if variable == 'temperature':

            # Check if temperature variable is not None
            if temp is None:
                raise ValueError("Temperature array required for temperature-based MLD")
            
            # Remove masked values
            depth_valid, temp_valid = valid_profile(temp)
            if depth_valid is None or len(depth_valid) < 2:
                return np.nan

            # Compute dT/dz using numpy.gradient
            grad = np.gradient(temp_valid, depth_valid)

            # First depth where gradient exceeds threshold gradT
            idx = np.where(np.ma.abs(grad) >= gradT)[0]

        elif variable == 'density':

            # Check if density variable is not None
            if density is None:
                raise ValueError("Density array required for density-based MLD")
            
            # Remove masked values
            depth_valid, dens_valid = valid_profile(density)
            if depth_valid is None or len(depth_valid) < 2:
                return np.nan

            # Compute dσθ/dz using numpy.gradient
            grad = np.gradient(dens_valid, depth_valid)

            # First depth where gradient exceeds threshold gradSigma
            idx = np.where(grad >= gradSigma)[0]

        else:
            raise ValueError("variable must be 'temperature' or 'density'")
    
    # ------------------------------
    # Potential Energy Anomaly method
    # ------------------------------

    elif method == 'potential_energy':
    
        # Check if density variable is not None
        if density is None:
            raise ValueError("Density array required for potential energy method")
        
        # Remove masked values
        depth_valid, dens_valid = valid_profile(density)
        if depth_valid is None or len(depth_valid) < 3:
            return np.nan
        
        # Compute the vertical spacing between depth levels and rename the density variable
        rho = dens_valid
        dz = np.diff(depth_valid)

        # Compute the surface to depth level layer thickness 
        H  = np.cumsum(dz) 

        # Compute the cumulative integral of rho(z) (estimate of the total mass per area of the water column from the surface down to each depth level)
        rho_m_int = np.cumsum(rho[:-1] * dz)

        # Compute the mean density of the layer from the surface to each depth (average density of the water column above each depth)
        rho_m = rho_m_int / H

        # Compute the local contribution to the potential energy anonaly at each depth
        integrand = (rho[:-1] - rho_m) * g * depth_valid[:-1]

        # Compute the total potential energy per unit area needed to mixed the fluid uniformly to each depth level
        pe_anomaly = np.cumsum(integrand * dz)

        # Compute the difference between the computed PEA and the target threshold 
        pea_diff = pe_anomaly - phi

        ###################
        # Note
        # ----
        # Breaking these lines of code down further, the difference between the average density of the water column above each depth 
        # and the density at each depth level
        # 
        # rho_m - rho(z) 
        # 
        # tells us how much lighter or denser the local water parcel is compared to the mean. By multiplying by g * z, we convert the
        # density anomaly to a potential energy per unit volume because
        # 
        # PE = rho * g * z (J / m^3)
        # 
        # Because rho_m is the average density of the water column above each depth level, this potential energy represent how much
        # pontential energy is required to homogenize the layer at each depth increment (by layer, we are refering to the layer of
        # fluid between each depth increment).  
        # 
        # Lastly, the mixed layer depth can be defined as the depth where the required mixing energy first exceeds a reference energy phi. 
        # 
        # PE_anomaly(z_mld) = phi
        # 
        # Choosing phi = 100 J/m^2 corresponds to a characteristic energetic "cost" to mix the upper layer that roughly separates weakly stratified
        # surface layers from strongly stratified pycnocline layers. 100 J/m^2 is an empiricla standard that has been proven to: 
        # 
        #  (1) Large enough to avoid noise from near-surface micro-stratification
        #  (2) Small enough to capture the physical mixed region. 
        #  
        #  Furthermore, the PEA method with phi = 100 J/m^2 have be shown to obtain similar results to the threshold methods. 
        ###################

        # Detect zero crossing (sign change)
        sign_change = np.sign(pea_diff[1:] * pea_diff[:-1]) < 0

        # Handle the case with no sign change (pe_anomaly does not exceed the threshold phi)
        if not np.any(sign_change):
            return np.nan
        
        # Obtain the first zero crossing index
        idx = np.argmax(sign_change)

        # Linear interpolation to find crossing
        mld = depth_valid[idx] - pea_diff[idx] * (depth_valid[idx+1] - depth_valid[idx]) / (pea_diff[idx+1] - pea_diff[idx])

    else:
        raise ValueError("method must be 'threshold', 'gradient', or 'potential_energy'")
        

    # ------------------------------
    # Return result
    # ------------------------------
    if method == 'potential_energy': 
        return mld 
    elif (method == 'threshold') | (method == 'gradient'):
        if idx.size == 0:
            # No depth satisfies the criterion → return NaN
            return np.nan
        else:
            # Return the shallowest depth where criterion is met
            return depth[idx[0]]

        
#--- Convert from lon,lat to easting,northing Function ---# 
def lonlat_to_xy_km(lon, lat, lon0=None, lat0=None, ellps="WGS84", return_vectors=False):
    """
    Convert longitude and latitude coordinates to Cartesian distances (km)
    relative to a specified or automatic origin.

    Parameters
    ----------
    lon : array_like
        Longitudes of data points (degrees). Can be 1D or 2D.
    lat : array_like
        Latitudes of data points (degrees). Must have same shape as lon.
    lon0 : float, optional
        Longitude of origin (degrees). If None, uses mean(lon).
    lat0 : float, optional
        Latitude of origin (degrees). If None, uses mean(lat).
    ellps : str, default='WGS84'
        Ellipsoid model used for great-circle distance calculations.
        Common options: 'WGS84', 'sphere', 'GRS80', etc.
    return_vectors : bool, default=False
        If True and input is 1D, also returns 1D coordinate vectors (x_cor, y_cor).

    Returns
    -------
    X, Y : ndarray
        Cartesian coordinates (km) relative to origin.
        X increases eastward, Y increases northward.
    lon0, lat0 : float
        Longitude and latitude of the chosen origin (degrees).
    x_cor, y_cor : 1D arrays, optional
        Only returned if `return_vectors=True` and input is 1D.

    Notes
    -----
    - Distances are computed geodesically using pyproj.Geod.
    - The coordinate origin (0, 0) is located at (lon0, lat0).
    - Sign convention:
        East of origin → +X
        North of origin → +Y
    - If `lon` and `lat` are 1D, outputs are 2D meshgrids created from them.
    """

    # Import libraries
    import numpy as np
    from pyproj import Geod

    # ---------------------------------------------------
    # 1. Initialize geodesic and input arrays
    # ---------------------------------------------------
    geod = Geod(ellps=ellps)
    lon = np.asarray(lon)
    lat = np.asarray(lat)

    # Validate inputs
    if lon.shape != lat.shape and not (lon.ndim == 1 and lat.ndim == 1):
        raise ValueError("lon and lat must be either 1D arrays of same length or 2D arrays of same shape.")

    # ---------------------------------------------------
    # 2. Determine the origin
    # ---------------------------------------------------
    if lon0 is None:
        lon0 = np.nanmean(lon)
    if lat0 is None:
        lat0 = np.nanmean(lat)

    # ---------------------------------------------------
    # 3. Handle 1D coordinate case
    # ---------------------------------------------------
    if lon.ndim == 1 and lat.ndim == 1:
        nx, ny = len(lon), len(lat)

        # Initialize coordinate vectors
        x_cor = np.zeros(nx)
        y_cor = np.zeros(ny)

        # --- Compute easting distances (longitude direction) ---
        for i in range(nx):
            # Compute geodesic distance from origin to (lon[i], lat0)
            _, _, dist = geod.inv(lon0, lat0, lon[i], lat0)
            # Convert to km and apply sign (west = negative)
            x_cor[i] = np.sign(lon[i] - lon0) * dist / 1000

        # --- Compute northing distances (latitude direction) ---
        for j in range(ny):
            # Compute geodesic distance from origin to (lon0, lat[j])
            _, _, dist = geod.inv(lon0, lat0, lon0, lat[j])
            # Convert to km and apply sign (south = negative)
            y_cor[j] = np.sign(lat[j] - lat0) * dist / 1000

        # --- Create 2D meshgrids ---
        X, Y = np.meshgrid(x_cor, y_cor)

        # Optionally return 1D vectors
        if return_vectors:
            return X, Y, lon0, lat0, x_cor, y_cor
        else:
            return X, Y, lon0, lat0

    # ---------------------------------------------------
    # 4. Handle 2D coordinate case (curvilinear grid)
    # ---------------------------------------------------
    elif lon.ndim == 2 and lat.ndim == 2:
        # Initialize output arrays
        X = np.zeros_like(lon, dtype=float)
        Y = np.zeros_like(lat, dtype=float)

        # Compute X and Y distances for each point
        for j in range(lat.shape[0]):
            for i in range(lon.shape[1]):
                # East-west distance (km)
                _, _, dist_x = geod.inv(lon0, lat0, lon[j, i], lat0)
                X[j, i] = np.sign(lon[j, i] - lon0) * dist_x / 1000
                # North-south distance (km)
                _, _, dist_y = geod.inv(lon0, lat0, lon0, lat[j, i])
                Y[j, i] = np.sign(lat[j, i] - lat0) * dist_y / 1000

        # Return 2D coordinate fields and origin
        return X, Y, lon0, lat0
    

#--- Rotating 2D fields ---# 
def rotate_field(data, lon, lat, theta_deg, dx_r = 1.5, dy_r = 1.5):
    """
    Rotate a 2D field clockwise by a given angle and crop to a square region.

    Parameters
    ----------
    data : 2D array (lat x lon)
        Input field to rotate.
    lon, lat : 1D arrays
        Longitude and latitude coordinates of the field.
    theta_deg : float
        Clockwise rotation angle in degrees.
    dx_r, dy_r : float
        Rotated coordinates' resolution in kilometers 

    Returns
    -------
    data_rot_crop : 2D array
        Rotated and cropped field (on a regular square grid in km).
    X_new, Y_new : 2D arrays
        Corresponding km-coordinate grids after rotation.
    """

    # Import libraries
    import numpy as np
    from pyproj import Geod
    from scipy.interpolate import griddata

    ###################################################
    # STEP 1: Convert lat/lon to distances (km) relative to grid center 
    ###################################################
    
    # Set geoid 
    geod = Geod(ellps="WGS84")

    # Set center longitude and latitude (origin of the transformed coordinate system) and the length of the coordinate vectors
    lon0, lat0 = np.mean(lon), np.mean(lat)
    nx, ny = len(lon), len(lat)

    # Initialize northing and easting coordinates
    x_cor = np.zeros(nx)
    y_cor = np.zeros(ny)

    # Loop through longtiude
    for i in range(nx):

        # Compute the distance from the origin 
        _, _, dist = geod.inv(lon0, lat0, lon[i], lat0)

        # Convert to kilometers and set the sign of x-coordinate vector (West of origin = negative)
        x_cor[i] = np.sign(lon[i] - lon0) * dist / 1000  # km

    # Loop through Latitude
    for j in range(ny):

        # Compute the distance from the origin
        _, _, dist = geod.inv(lon0, lat0, lon0, lat[j])

        # Convert to kilometers and set the sign of y-coordinate vector (SOuth of origin = negative)
        y_cor[j] = np.sign(lat[j] - lat0) * dist / 1000  # km

    # Create a mesh grid 
    X, Y = np.meshgrid(x_cor, y_cor)

    ###################################################
    # STEP 2: Rotate coordinate system clockwise 
    ###################################################
    
    # Convert the angle to radians
    theta = np.deg2rad(theta_deg)

    # Rotate coordinate vectors
    Xr =  X * np.cos(theta) + Y * np.sin(theta)
    Yr = -X * np.sin(theta) + Y * np.cos(theta)

    # Flatten arrays
    Xr_flat = Xr.flatten()
    Yr_flat = Yr.flatten()
    data_flat = data.flatten()

    ###################################################
    # STEP 3: Crop data within largest inscribed square 
    ###################################################

    # Set half-width of the biggest square you can draw inside the rotated data region
    crop_half = largest_centered_square_halfwidth(Xr, Yr)

    # Obtain indices of cropped region
    idx_x = (Xr_flat >= -crop_half) & (Xr_flat <= crop_half)
    idx_y = (Yr_flat >= -crop_half) & (Yr_flat <= crop_half)
    idx = np.logical_and(idx_x, idx_y)

    # Apply cropping mask
    xr_crop, yr_crop, data_crop = Xr_flat[idx], Yr_flat[idx], data_flat[idx]

    ###################################################
    # Step 4: Interpolate cropped region onto a regular grid
    ###################################################
    
    # Set your grid extent
    xr_min, xr_max = xr_crop.min(), xr_crop.max()
    yr_min, yr_max = yr_crop.min(), yr_crop.max()

    # Define new grid 
    xr_grid = np.arange(xr_min, xr_max + dx_r, dx_r) 
    yr_grid = np.arange(yr_min, yr_max + dy_r, dy_r)

    # Set the meshgrid
    XG, YG = np.meshgrid(xr_grid, yr_grid)  # shape (Ny, Nx)

    # Flatten coordinates
    points = np.stack([xr_crop, yr_crop], axis=-1)  # shape (N, 2)

    # Interpolate SST onto grid
    data_grid = griddata(
        points, data_crop, (XG, YG),
        method='linear'  # or 'cubic', or 'nearest'
    )

    ###################################################
    # STEP 5: Trim NaN values at the edges of the interpolated grid
    ###################################################
    
    # Define a mask where NaNs = 0, Valid points = 1
    mask = ~np.isnan(data_grid)

    # Find the rows and columns with at least one valid point
    row_inds = np.where(mask.any(axis=1))[0]
    col_inds = np.where(mask.any(axis=0))[0]

    # Trim the coordinates and data 
    data_trim = data_grid[row_inds[0]:row_inds[-1]+1, col_inds[0]:col_inds[-1]+1]
    XG_trim = XG[row_inds[0]:row_inds[-1]+1, col_inds[0]:col_inds[-1]+1]
    YG_trim = YG[row_inds[0]:row_inds[-1]+1, col_inds[0]:col_inds[-1]+1]

    # # Define another mask with trimmed array  
    # mask = ~np.isnan(data_trim)

    # # Compute the fraction of valid points in across rows and columns
    # row_valid_frac = mask.mean(axis=1)
    # col_valid_frac = mask.mean(axis=0)

    # # Find rows and columns that have at least 95% filled with valid points
    # valid_rows = np.where(row_valid_frac > 0.95)[0]  
    # valid_cols = np.where(col_valid_frac > 0.95)[0]

    # # Trim array 
    # if len(valid_rows) > 0 and len(valid_cols) > 0:
    #     row_min, row_max = valid_rows[0], valid_rows[-1]
    #     col_min, col_max = valid_cols[0], valid_cols[-1]

    #     data_n = data_trim[row_min:row_max+1, col_min:col_max+1]
    #     X_n = XG_trim[row_min:row_max+1, col_min:col_max+1]
    #     Y_n = YG_trim[row_min:row_max+1, col_min:col_max+1]
    # else:
    #     raise ValueError("No fully valid (non-NaN) region found in data.")
    
    # # Reset the origin 
    # X_n = X_n - np.min(X_n)
    # Y_n = Y_n - np.min(Y_n)

    # Reset the origin 
    XG_trim = XG_trim - np.min(XG_trim)
    YG_trim = YG_trim - np.min(YG_trim)

    return data_trim, XG_trim, YG_trim #data_n, X_n, Y_n

#--- Trimming corners after rotation function ---# 
def largest_centered_square_halfwidth(Xr, Yr, shrink_eps=1e-6, iters=32):

    """
    Compute the half-width (crop_half) of the largest axis-aligned square,
    centered at (0,0), that fits entirely within the convex hull of the
    rotated grid points (Xr, Yr).

    Parameters
    ----------
    Xr, Yr : 2D arrays
        Rotated x and y coordinates (km).
    shrink_eps : float, optional
        Small fractional shrink factor to avoid floating-point boundary issues.
        The final half-width is reduced by (1 - shrink_eps).
    iters : int, optional
        Number of bisection iterations (more = finer precision).

    Returns
    -------
    crop_half : float
        Half-width (km) of the largest centered square that fits within
        the convex hull of (Xr, Yr).
    """

    # Import libraries
    from scipy.spatial import ConvexHull
    from matplotlib.path import Path
    import numpy as np

    # --- Step 1: Flatten all rotated coordinates into N×2 array of points ---
    pts = np.column_stack((Xr.ravel(), Yr.ravel()))

    # --- Step 2: Compute convex hull enclosing all rotated grid points ---
    # The convex hull is the smallest polygon containing all data points.
    hull = ConvexHull(pts)

    # Extract the ordered vertices of that polygon
    poly = pts[hull.vertices]

    # --- Step 3: Create a Path object for efficient point-in-polygon testing ---
    path = Path(poly)

    # --- Step 4: Define upper and lower bounds for half-width search ---
    # Start with the half-length of the smallest bounding-box side as the max possible.
    x_min, x_max = Xr.min(), Xr.max()
    y_min, y_max = Yr.min(), Yr.max()
    hi = min((x_max - x_min), (y_max - y_min)) * 0.5
    lo = 0.0  # lower bound starts at zero

    # --- Step 5: Use bisection search to find the largest inscribed square ---
    # The algorithm repeatedly tests if a square of half-width h is fully inside the hull.
    for _ in range(iters):

        # Test the midpoint of the current [lo, hi] range
        h = 0.5 * (lo + hi)

        # Define the four corners of a centered square with half-width h
        corners = np.array([[ h,  h],
                            [ h, -h],
                            [-h,  h],
                            [-h, -h]])

        # Check whether all four corners lie inside the convex hull polygon
        inside = path.contains_points(corners).all()

        # If all corners are inside, we can expand the square (move lower bound up)
        if inside:
            lo = h
        # Otherwise, shrink the candidate square (move upper bound down)
        else:
            hi = h

    # --- Step 6: Apply a tiny shrink factor to guarantee it's strictly inside ---
    crop_half = lo * (1.0 - shrink_eps)

    return crop_half


#--- Cartesian to polar coordinates transformation ---# 
def lonlat_to_polar_planar(lon, lat):
    
    """
    Transform (lon, lat) coordinates to (r, theta) polar coordinates relative
    to the grid center, using a planar (local tangent-plane) approximation.

    Parameters
    ----------
    lon : 2D array
        Longitude grid (degrees).
    lat : 2D array
        Latitude grid (degrees). Must have same shape as `lon`.

    Returns
    -------
    r : 2D array
        Radial distance from the grid center (km).
    theta : 2D array
        Polar angle (degrees), measured clockwise from north.
    """

    # Import library
    import numpy as np

    # --- Constants --- #
    R_earth = 6371.0  # Earth's mean radius [km]

    # --- Compute grid center --- #
    lon0 = np.mean(lon)
    lat0 = np.mean(lat)

    # --- Convert degrees to kilometers --- #
    deg2km_lat = (np.pi / 180.0) * R_earth                # km per degree latitude
    deg2km_lon = deg2km_lat * np.cos(np.deg2rad(lat0))    # km per degree longitude (adjust for latitude)

    # --- Compute local Cartesian offsets (dx eastward, dy northward) --- #
    dx = (lon - lon0) * deg2km_lon   # east–west distance in km
    dy = (lat - lat0) * deg2km_lat   # north–south distance in km

    # --- Polar coordinates --- #
    r = np.sqrt(dx**2 + dy**2)
    # theta: 0° = north, increases clockwise
    theta = (np.degrees(np.arctan2(dx, dy)) + 360.0) % 360.0

    return r, theta

# =====================================================================
# Function: interpolate_to_polar_grid
# =====================================================================
def interpolate_to_polar_grid(lon, lat, data, lon_center, lat_center):
    """
    Interpolate a 2D field from Cartesian (lon, lat) to a regular polar grid.
    The radial (dr) and angular (dtheta) resolutions are automatically chosen
    to match the average resolution of the input Cartesian grid.
    The output is cropped to the largest inscribed circle (no NaNs outside).
    The polar angle follows the *geographic convention*: reference = North,
    increasing clockwise (θ = 0° → North, 90° → East).

    Parameters
    ----------
    lon, lat : 2D arrays
        Longitude and latitude grids (same shape).
    data : 2D array
        Data values on the (lon, lat) grid.
    lon_center, lat_center : float
        Center point for the polar coordinate transformation.

    Returns
    -------
    R : 2D array
        Radial coordinates (cropped to inscribed circle).
    THETA : 2D array
        Angular coordinates (radians, 0 = North, increasing clockwise).
    data_polar : 2D array
        Interpolated data on the polar grid.
    r_inscribed : float
        Radius of the largest inscribed circle (for reference).
    """

    # Import libraries
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.interpolate import griddata

    # ---------------------------------------------------------
    # 1. Convert (lon, lat) grid to local Cartesian coordinates
    # ---------------------------------------------------------
    # Use a planar (flat-Earth) approximation centered on (lon_center, lat_center)
    # 1° latitude ≈ 111 km; longitude scaled by cos(latitude)
    x = (lon - lon_center) * np.cos(np.deg2rad(lat_center)) * 111.0  # km
    y = (lat - lat_center) * 111.0                                   # km

    # ---------------------------------------------------------
    # 2. Determine grid spacing and domain limits
    # ---------------------------------------------------------
    # Compute approximate resolution in x and y directions
    dx = np.nanmean(np.diff(x, axis=1))
    dy = np.nanmean(np.diff(y, axis=0))
    dr = np.mean([dx, dy])  # Representative spatial resolution (km)

    # Compute domain bounds and the radius of the largest inscribed circle
    # The inscribed circle fits fully within the rectangular grid
    x_min, x_max = np.nanmin(x), np.nanmax(x)
    y_min, y_max = np.nanmin(y), np.nanmax(y)
    r_inscribed = min((x_max - x_min) / 2, (y_max - y_min) / 2)

    # ---------------------------------------------------------
    # 3. Define polar grid based on input resolution
    # ---------------------------------------------------------
    # (a) Define the radial coordinate array
    #     - The maximum radius is limited to the inscribed circle.
    #     - The number of bins is set so that Δr ≈ native grid spacing.
    n_r = int(np.ceil(r_inscribed / dr))
    r_bins = np.linspace(0, r_inscribed, n_r)

    # (b) Define the angular coordinate array
    #     - Choose angular spacing so the arc length at r_inscribed ≈ dr.
    #     - This keeps approximately uniform sampling density in both directions.
    dtheta = dr / max(r_inscribed, 1e-6)  # radians per bin
    n_theta = int(np.ceil(2 * np.pi / dtheta))
    theta_bins = np.linspace(0, 2 * np.pi, n_theta)

    # (c) Create the full polar grid
    R, THETA = np.meshgrid(r_bins, theta_bins)

    # ---------------------------------------------------------
    # 4. Interpolate data from (x, y) → (R, THETA)
    # ---------------------------------------------------------
    # (a) Convert from polar (R, THETA) to Cartesian (X, Y)
    #     Geographic convention: θ=0°→North, increase clockwise
    X = R * np.sin(THETA)   # East–West (x)
    Y = R * np.cos(THETA)   # North–South (y)

    # (b) Prepare the original data points for interpolation
    points = np.column_stack((x.ravel(), y.ravel()))
    values = data.ravel()

    # (c) Perform interpolation using linear method
    data_polar = griddata(points, values, (X, Y), method='linear')

    # (d) Optionally trim grid edges beyond valid interpolation region
    #     (This avoids empty radial rings due to rectangular boundary geometry)
    valid_mask = np.isfinite(data_polar)
    if np.any(valid_mask):
        # Find last radial index with valid data in any θ
        max_valid_r_index = np.max(np.where(np.any(valid_mask, axis=0)))
        R = R[:, :max_valid_r_index + 1]
        THETA = THETA[:, :max_valid_r_index + 1]
        data_polar = data_polar[:, :max_valid_r_index + 1]

    return R, THETA, data_polar, r_inscribed


import numpy as np
from scipy.interpolate import interp1d
from numpy.linalg import eig


#--- Baroclinic Rossby wave vertical modes function ---#
def moderossby(zin, nin, lat, nmodes=20, nz=1024):
    """
    Solve the vertical mode equation for baroclinic Rossby waves with a free surface.

    This function computes the first `nmodes` baroclinic modes by solving the
    eigenvalue problem

        d²W/dz² + λ² (N² / f²) W = 0,

    with a free-surface boundary condition at z = 0 and rigid bottom at z = D.
    The eigenvalues λ² are related to the horizontal scales via

        λ² = -(k² + l² + β k / σ),

    and 1/|λ| corresponds to a Rossby deformation radius.

    Parameters
    ----------
    zin : array_like, shape (n_z_in,)
        Input depth grid (e.g., from data), typically from surface to bottom [m].
    nin : array_like, shape (n_z_in,)
        Buoyancy frequency N (called nin here) on `zin`, in radians per second.
        Any zero values are replaced by the local Coriolis frequency `f`.
    lat : float
        Latitude in degrees.
    nmodes : int, optional
        Number of baroclinic modes to return (default is 20).
    nz : int, optional
        Number of vertical grid points used for the internal discretization.
        Default is 1024, matching the MATLAB code.

    Returns
    -------
    lambda1sq : ndarray, shape (nmodes,)
        The squared eigenvalues λ² for the selected modes.
    w1 : ndarray, shape (nz+1, nmodes)
        Vertical mode shapes W(z), normalized so that W(0) = 1 for each mode.
        The last level corresponds to the bottom and is set to zero (rigid bottom).
    z1 : ndarray, shape (nz+1,)
        Depth grid corresponding to `w1` [m], from 0 (surface) to D (bottom).
    w0 : ndarray, shape (nz+1,)
        Barotropic vertical mode W(z), normalized so W(0)=1.
    """
    zin = np.asarray(zin, dtype=float)
    nin = np.asarray(nin, dtype=float)

    # -------------------------------------------------------------------------
    # 1. Basic parameters: Coriolis, gravity, etc.
    # -------------------------------------------------------------------------
    # Convert latitude to radians
    phi = np.deg2rad(lat)

    # Earth's rotation rate (1/s)
    fe = 2.0 * np.pi / (24.0 * 3600.0)

    # Coriolis parameter
    f = 2.0 * fe * np.sin(phi)

    # Gravity
    grav = 9.8

    # Replace any zero N with f (same as MATLAB: nin(indx) = f)
    nin = nin.copy()
    nin[nin == 0] = f

    # -------------------------------------------------------------------------
    # 2. Vertical grid and interpolation of N
    # -------------------------------------------------------------------------
    # Bottom depth
    h = np.max(zin)

    # Uniform vertical spacing (same logic as MATLAB: dz = h / nz; z = 0:dz:h-dz)
    dz = h / nz
    z = np.linspace(0.0, h - dz, nz)  # internal grid for the eigenproblem

    # Interpolate N onto this uniform grid (cubic spline ~ 'spline' in MATLAB)
    f_interp = interp1d(zin, nin, kind='cubic', fill_value='extrapolate')
    n = f_interp(z)

    # g in the code is actually (f^2 / N^2), used as a weighting factor
    g_weight = f**2 / (n**2)

    fact = 1.0 / dz**2

    # -------------------------------------------------------------------------
    # 3. Build the finite-difference matrix A for the eigenvalue problem
    #
    #    A W = λ² W,  where λ² are (minus) the eigenvalues of A as constructed.
    # -------------------------------------------------------------------------
    A = np.zeros((nz, nz), dtype=float)

    # --- Free-surface boundary condition at the top (z = 0) ---
    #
    #  dW/dz(0) = (g λ² / f²) W(0)
    #
    # Discretized with a forward difference:
    #  dW/dz(0) ~ (W_2 - W_1) / dz
    #
    # This gets baked into the matrix as the first row.
    A[0, 0] = -(f**2 / grav) * np.sqrt(fact)   # -(f^2/g)*(1/dz)
    A[0, 1] =  (f**2 / grav) * np.sqrt(fact)   #  (f^2/g)*(1/dz)

    # --- Rigid-bottom boundary condition at the bottom (z = h) ---
    #
    #  W(D) = 0  → implemented through the last row of the finite-diff operator.
    #
    A[-1, -2] = g_weight[-1] * fact
    A[-1, -1] = -2.0 * g_weight[-1] * fact

    # --- Interior second-derivative operator ---
    #
    # For 2 <= j <= nz-1:
    #   d²W/dz² ~ (W_{j+1} - 2 W_j + W_{j-1}) / dz²
    #
    for j in range(1, nz - 1):
        A[j, j]     = -2.0 * g_weight[j] * fact
        A[j, j + 1] =  g_weight[j] * fact
        A[j, j - 1] =  g_weight[j] * fact

    # -------------------------------------------------------------------------
    # 4. Solve eigenvalue problem A v = d v
    # -------------------------------------------------------------------------
    # eigvals, eigvecs such that A v_k = d_k v_k
    eigvals, eigvecs = eig(A)

    # MATLAB does: vals = -diag(d)
    vals = -eigvals

    # Sort eigenvalues by increasing |λ²| (like MATLAB: [yy, ii] = sort(abs(vals)))
    idx_sorted = np.argsort(np.abs(vals))
    lambdasq_sorted = vals[idx_sorted].real

    # -------------------------------------------------------------------------
    # 5. Extract nmodes modes
    # -------------------------------------------------------------------------
    nmodes = int(nmodes)
    if nmodes + 1 > len(vals):
        raise ValueError("Requested more modes than available eigenvalues.")

    # --- Barotropic (external) mode: index 0 in sorted list ---# 

    # Index of the barotropic mode (smallest |λ²| in sorted list)
    eig_index_bt = idx_sorted[0]

    # Extract barotropic eigenvector (take real part to remove tiny imag noise)
    mode_bt = eigvecs[:, eig_index_bt].real

    # Normalize so surface value W(0) = 1
    mode_bt = mode_bt / mode_bt[0]

    # Build full vertical mode including bottom boundary
    w0 = np.zeros(nz + 1)
    w0[:-1] = mode_bt      # interior levels
    w0[-1] = 0.0           # rigid bottom: W(D) = 0

    # Barotropic eigenvalue λ²
    lambda0sq = lambdasq_sorted[0]

    # --- Baroclinic internal modes: 1 to nmodes (skipping the first) ---#
    lambda1sq = np.empty(nmodes, dtype=float)
    w1 = np.zeros((nz + 1, nmodes), dtype=float)  # +1 to include bottom with W=0

    for j in range(nmodes):
        # Index of the j-th desired mode in the sorted list (skip the first one)
        eig_index = idx_sorted[j + 1]

        # Corresponding eigenvector
        mode = eigvecs[:, eig_index].real  # take real part; eigenvectors may be complex

        # Normalize so that surface value is 1 (W(0) = 1)
        mode = mode / mode[0]

        # Store in w1: first nz entries from the eigenvector, last entry is 0 (bottom)
        w1[:-1, j] = mode
        w1[-1, j] = 0.0

        # Store squared eigenvalue
        lambda1sq[j] = lambdasq_sorted[j + 1].real

    # Depth grid for modes: from 0 to h inclusive, nz+1 points (like z1 = 0:dz:h)
    z1 = np.linspace(0.0, h, nz + 1)

    return lambda1sq, w1, z1, lambda0sq, w0



