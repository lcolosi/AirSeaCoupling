# California Current Ecosystem Mooring Processing functions
## Luke Colosi | lcolosi@ucsd.edu 

#--- Rotate and bin-average velocity ---# 
def rotate_cce_vel(u, v, time, depth, bathy_grid, lon_grid, lat_grid, option_plot, 
                   isobath, x_bnds, y_bnds, idx_site, option_mask, depth_thresh):

    """
    Function for rotating velocities into along-shelf and cross-shelf coordinates
    defined by a specified isobath, and computing depth-averaged and 12-hour binned
    velocities at a chosen site.

    Parameters
    ----------
    u : numpy.ma.MaskedArray
        Zonal velocity at all sites, times, and depths.
        Shape (n_site, n_time, n_depth). Units typically m s^-1.
    v : numpy.ma.MaskedArray
        Meridional velocity at all sites, times, and depths.
        Shape (n_site, n_time, n_depth). Units typically m s^-1.
    time : array-like of datetime.datetime
        Time stamps for the second dimension of `u` and `v` (length n_time).
    depth : array-like
        Vertical coordinate for the third dimension of `u` and `v`
        (length n_depth). Typically meters, negative downward.
    bathy_grid : 2D array-like
        Bathymetry (depth in meters) on the lon/lat grid (same shape as `lon_grid`).
    lon_grid : 2D array-like
        Longitudes of the bathymetry grid. Degrees East.
    lat_grid : 2D array-like
        Latitudes of the bathymetry grid. Degrees North.
    option_plot : bool or int
        If True/1, show diagnostic plots for isobath extraction and fitting.
    isobath : float
        Target depth (m) used to define the isobath (e.g., -1500).
    x_bnds : array-like of float, length 2
        Easting bounds (km, internally centered/scaled) used to select the
        isobath segment for the linear fit: [x_min, x_max].
    y_bnds : array-like of float, length 2
        Northing bounds (km, internally centered/scaled) used to select the
        isobath segment for the linear fit: [y_min, y_max].
    idx_site : int
        Index of the site along the first dimension of `u` and `v`.
    option_mask : bool or int
        If True/1, depth-averaging only uses depths shallower than `depth_thresh`.
        If False/0, all depths are used.
    depth_thresh : float
        Maximum depth (m, positive) used for depth-averaging when `option_mask` is True.

    Returns
    -------
    u_bar : numpy.ma.MaskedArray, shape (n_time,)
        Depth-averaged zonal velocity at native time resolution.
    v_bar : numpy.ma.MaskedArray, shape (n_time,)
        Depth-averaged meridional velocity at native time resolution.
    u_bar_bin : numpy.ndarray, shape (n_bin,)
        12-hour binned depth-averaged zonal velocity.
    v_bar_bin : numpy.ndarray, shape (n_bin,)
        12-hour binned depth-averaged meridional velocity.
    u_along : numpy.ma.MaskedArray, shape (n_time, n_depth)
        Along-shelf velocity at native time resolution. Positive upcoast
        (approximately northward).
    v_cross : numpy.ma.MaskedArray, shape (n_time, n_depth)
        Cross-shelf velocity at native time resolution. Positive offshore
        (approximately westward).
    u_along_bin : numpy.ma.MaskedArray, shape (n_bin, n_depth)
        12-hour binned along-shelf velocity profile.
    v_cross_bin : numpy.ma.MaskedArray, shape (n_bin, n_depth)
        12-hour binned cross-shelf velocity profile.
    u_along_bar : numpy.ma.MaskedArray, shape (n_time,)
        Depth-averaged along-shelf velocity at native time resolution.
    v_cross_bar : numpy.ma.MaskedArray, shape (n_time,)
        Depth-averaged cross-shelf velocity at native time resolution.
    u_along_bar_bin : numpy.ndarray, shape (n_bin,)
        12-hour binned depth-averaged along-shelf velocity.
    v_cross_bar_bin : numpy.ndarray, shape (n_bin,)
        12-hour binned depth-averaged cross-shelf velocity.
    theta_n : float
        Rotation angle (radians) defining the along-/cross-shelf axes.
    time_bin_dt : numpy.ndarray of datetime.datetime
        Time stamps for the centers of the 12-hour bins.

    Notes
    -----
    - The isobath is extracted from `bathy_grid` via contouring at `isobath`.
    - Coordinates are projected from (lon, lat) to UTM Zone 10N (EPSG:32610),
      then centered and scaled to km.
    - A straight-line fit to the isobath segment within `x_bnds` and `y_bnds`
      defines the local shelf orientation used for rotation.
    """
    # function body...


    # Import libraries
    import numpy as np
    import matplotlib.pyplot as plt 
    from pyproj import Proj, Transformer
    from datetime import timedelta, datetime

    #------------------------------------------# 
    # STEP #1 - Extract the isobath contour coordinates
    #------------------------------------------# 

    # Use contouring to extract the 1500 m isobath(s)
    contour_set = plt.contour(lon_grid, lat_grid, bathy_grid, levels=[isobath])
    plt.close()  # Prevents plot from displaying if you're in a notebook

    # Extract all contour paths at the 1500 m level
    paths = contour_set.allsegs[0] 

    # Filter based on number of points (to exclude small closed loops like seamounts)
    min_points = 100  # Adjust based on your resolution
    valid_paths = [p for p in paths if len(p) > min_points]

    # Select the longest or first valid path (shelf break usually shows up first)
    if isobath >= -500:
        selected_path = valid_paths[1]  # Or choose based on location
    else: 
        selected_path = valid_paths[0]  # Or choose based on location

    # Extract lon/lat from the path
    lon_iso = selected_path[:, 0]
    lat_iso = selected_path[:, 1]

    # Plot the -2000 meter isobath contours and selected path
    if option_plot == True:

        # Loop through paths
        for i, p in enumerate(paths):

            # Plot the ith contour path
            plt.plot(p[:, 0], p[:, 1], '.', label=f'Path {i}')

        # Plot the selected path 
        plt.plot(lon_iso,lat_iso, 'k-', label='Selected Path')

        # Set figure attributes
        plt.axis('equal')
        plt.grid(True,linestyle='--')
        plt.legend()
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.title("Extracted" + str(isobath) + "Isobath Paths")
        plt.show()

    #------------------------------------------# 
    # STEP #2 - Convert the isobath coordinates to easting and northing coordinates
    #------------------------------------------# 

    # Easting and Northing projection for the region, e.g., UTM zone 10N for California
    transformer = Transformer.from_crs("epsg:4326", "epsg:32610", always_xy=True)  # WGS84 to UTM Zone 10N

    # Convert lon/lat to x/y in meters
    x_iso, y_iso = transformer.transform(lon_iso, lat_iso)

    # Adjust isobath coordinates
    x_iso_n, y_iso_n = (1/1000)* (x_iso - np.mean(x_iso)), (1/1000)* (y_iso - np.mean(y_iso)) 

    # Plot the smoothed isobath in easting and northing coordinates
    if option_plot == True:

        # Plot the unsmoothed and smoothed isobaths 
        plt.plot(x_iso_n,y_iso_n, 'k-', label= str(isobath) + ' meter Isobath')

        # Set figure attributes
        plt.axis('equal')
        plt.grid(True,linestyle='--')
        plt.legend()
        plt.xlabel("Easting")
        plt.ylabel("Northing")
        plt.show()

    #------------------------------------------# 
    # STEP #3 - Calculate the Orientation Angle for the isobath 
    #------------------------------------------# 

    # Mask easting and northing vectors based on limits above
    x_mask = (x_iso_n >= x_bnds[0]) & (x_iso_n <= x_bnds[1]) 
    y_mask = (y_iso_n >= y_bnds[0]) & (y_iso_n <= y_bnds[1]) 
    mask = x_mask & y_mask 

    # Apply mask
    x_iso_mask = x_iso_n[mask]
    y_iso_mask = y_iso_n[mask]

    # Fit a straight line to the bathymetry
    m, b = np.polyfit(x_iso_mask, y_iso_mask, 1)  

    # Compute the angle
    theta = np.arctan(m)  

    # Orient the angle so positive along-shelf is upcoast (northernly) and positive cross-shelf is offshore (westerly) 
    theta_n = theta - (np.pi)

    # Plot the linear fit and print the rotation angle 
    if option_plot == True:

        # Print the angle in units of degrees
        print(np.rad2deg(theta) % 360)
        print(np.rad2deg(theta_n) % 360)

        # Compute the linear fit line 
        bfit = m*(x_iso_mask) + b

        # Plot the isobath and linear fit 
        plt.plot(x_iso_n,y_iso_n, 'k-', label= str(isobath) + ' Isobath')
        plt.plot(x_iso_mask,y_iso_mask, 'r-', label = ' Fitting section of Isobath')
        plt.plot(x_iso_mask,bfit, 'b-', label='Linear LSF')

        # Set figure attributes
        plt.axis('equal')
        plt.grid(True,linestyle='--')
        plt.legend()
        plt.xlabel("Easting (km)")
        plt.ylabel("Northing (km)")
        plt.show()

    #------------------------------------------# 
    # STEP #4 - Compute the depth averaged velocity 
    #------------------------------------------# 

    # Grab data from the desired site
    u_site = u[idx_site,:,:]
    v_site = v[idx_site,:,:]

    #--- Compute weigthed depth average ---# 

    # Take absolute value of depth
    depth_pos = np.abs(depth)

    # Ensure increasing order (required by np.trapz)
    if not np.all(np.diff(depth_pos) > 0):
        depth_pos = depth_pos[::-1]
        u_site = u_site[:, ::-1]
        v_site = v_site[:, ::-1]

    # Mask depth levels below a threshold
    if option_mask == 1: 

        # Mask depth levels deeper than depth_thresh
        mask = depth_pos <= depth_thresh

        # Select shallower depths
        depth_sel = depth_pos[mask]
        u_sel = u_site[:, mask]
        v_sel = v_site[:, mask]

        # Depth range
        H = depth_sel[-1] - depth_sel[0]

        # Weighted average
        u_bar_tmp = np.trapz(u_sel.filled(np.nan), depth_sel, axis=1) / H
        v_bar_tmp = np.trapz(v_sel.filled(np.nan), depth_sel, axis=1) / H

        # Convert arrays back to masked arrays
        u_bar = np.ma.masked_invalid(u_bar_tmp)
        v_bar = np.ma.masked_invalid(v_bar_tmp)

    else: 

        # Depth range
        H = depth_pos[-1] - depth_pos[0]

        # Weighted average
        u_bar_tmp = np.trapz(u_site.filled(np.nan), depth_pos, axis=1) / H
        v_bar_tmp = np.trapz(v_site.filled(np.nan), depth_pos, axis=1) / H

        # Convert arrays back to masked arrays
        u_bar = np.ma.masked_invalid(u_bar_tmp)
        v_bar = np.ma.masked_invalid(v_bar_tmp)

    #------------------------------------------# 
    # STEP #5 - Compute 12-hourly bin averaged velocity data (depth-averaged and depth-dependent) 
    #------------------------------------------# 

    # Set bin window length and convert to seconds
    bin_len_hr = 12                         # Units: hours
    bin_len_sec = bin_len_hr * 3600         # Units: seconds

    # Set the time elapsed from t0
    t0 = time[0]
    time_elapsed_hours = np.array([(t - t0).total_seconds() for t in time])

    # Assign time steps to bins
    bin_index = np.floor(time_elapsed_hours / bin_len_sec).astype(int)
    unique_bins = np.unique(bin_index)

    # Bin-average time and data 
    time_bin  = np.array([np.mean(time_elapsed_hours[bin_index == b]) for b in unique_bins])
    u_bar_bin = np.array([np.mean(u_bar[bin_index == b]) for b in unique_bins])
    v_bar_bin = np.array([np.mean(v_bar[bin_index == b]) for b in unique_bins])
    u_bin     = np.array([np.mean(u_site[bin_index == b,:], axis=0) for b in unique_bins])
    v_bin     = np.array([np.mean(v_site[bin_index == b,:], axis=0) for b in unique_bins])

    # Convert back to datetme
    time_tmp = np.array([t0 + timedelta(seconds=s) for s in time_bin])
    time_bin_dt = np.array([datetime(d.year, d.month, d.day, d.hour, d.minute, d.second) for d in time_tmp])

    #------------------------------------------# 
    # STEP #6 - Rotate velocity vectors 
    #------------------------------------------# 

    # Compute Rotation matrix components
    cos_a = np.cos(theta_n)
    sin_a = np.sin(theta_n)

    #--- Depth-dependent Velocities ---# 
    u_along_tmp = u_site * cos_a + v_site * sin_a
    v_cross_tmp = -u_site * sin_a + v_site * cos_a

    u_along_bin_tmp = u_bin * cos_a + v_bin * sin_a
    v_cross_bin_tmp = -u_bin * sin_a + v_bin * cos_a

    # Mask zeros
    u_along = np.ma.masked_where(u_along_tmp == 0, u_along_tmp)
    v_cross = np.ma.masked_where(v_cross_tmp == 0, v_cross_tmp)
    u_along_bin = np.ma.masked_where(u_along_bin_tmp == 0, u_along_bin_tmp)
    v_cross_bin = np.ma.masked_where(v_cross_bin_tmp == 0, v_cross_bin_tmp)

    #--- Depth-averaged Velocities ---# 
    u_along_bar = u_bar * cos_a + v_bar * sin_a
    v_cross_bar = -u_bar * sin_a + v_bar * cos_a

    u_along_bar_bin = u_bar_bin * cos_a + v_bar_bin * sin_a
    v_cross_bar_bin = -u_bar_bin * sin_a + v_bar_bin * cos_a

    return u_bar, v_bar, u_bar_bin, v_bar_bin,  u_along, v_cross, u_along_bin, v_cross_bin, u_along_bar, v_cross_bar, u_along_bar_bin, v_cross_bar_bin, theta_n,time_bin_dt
    
#--- Rotate velocities along a transect ---# 
def rotate_transect_vel(u, v, depth, bathy_grid, lon_grid, lat_grid, option_plot, 
                        isobath, x_bnds, y_bnds, option_mask, depth_thresh):

    """
    Function for rotating velocities into along-shelf and cross-shelf coordinates
    defined by a specified isobath, and computing depth-averaged
    velocities along a transect.

    Parameters
    ----------
    u : numpy.ma.MaskedArray
        Zonal velocity at all distances (along transect), times, and depths.
        Shape (n_dist, n_time, n_depth). Units typically m s^-1.
    v : numpy.ma.MaskedArray
        Meridional velocity at all distances (along transect), times, and depths.
        Shape (n_dist, n_time, n_depth). Units typically m s^-1.
    depth : array-like
        Vertical coordinate for the third dimension of `u` and `v`
        (length n_depth). Typically meters, negative downward.
    bathy_grid : 2D array-like
        Bathymetry (depth in meters) on the lon/lat grid (same shape as `lon_grid`).
    lon_grid : 2D array-like
        Longitudes of the bathymetry grid. Degrees East.
    lat_grid : 2D array-like
        Latitudes of the bathymetry grid. Degrees North.
    option_plot : bool or int
        If True/1, show diagnostic plots for isobath extraction and fitting.
    isobath : float
        Target depth (m) used to define the isobath (e.g., -1500).
    x_bnds : array-like of float, length 2
        Easting bounds (km, internally centered/scaled) used to select the
        isobath segment for the linear fit: [x_min, x_max].
    y_bnds : array-like of float, length 2
        Northing bounds (km, internally centered/scaled) used to select the
        isobath segment for the linear fit: [y_min, y_max].
    option_mask : bool or int
        If True/1, depth-averaging only uses depths shallower than `depth_thresh`.
        If False/0, all depths are used.
    depth_thresh : float
        Maximum depth (m, positive) used for depth-averaging when `option_mask` is True.

    Returns
    -------
    u_bar : numpy.ma.MaskedArray, shape (n_time,)
        Depth-averaged zonal velocity at native time resolution.
    v_bar : numpy.ma.MaskedArray, shape (n_time,)
        Depth-averaged meridional velocity at native time resolution.
    u_along : numpy.ma.MaskedArray, shape (n_time, n_depth)
        Along-shelf velocity at native time resolution. Positive upcoast
        (approximately northward).
    v_cross : numpy.ma.MaskedArray, shape (n_time, n_depth)
        Cross-shelf velocity at native time resolution. Positive offshore
        (approximately westward).
    u_along_bar : numpy.ma.MaskedArray, shape (n_time,)
        Depth-averaged along-shelf velocity at native time resolution.
    v_cross_bar : numpy.ma.MaskedArray, shape (n_time,)
        Depth-averaged cross-shelf velocity at native time resolution.
    theta_n : float
        Rotation angle (radians) defining the along-/cross-shelf axes.

    Notes
    -----
    - The isobath is extracted from `bathy_grid` via contouring at `isobath`.
    - Coordinates are projected from (lon, lat) to UTM Zone 10N (EPSG:32610),
      then centered and scaled to km.
    - A straight-line fit to the isobath segment within `x_bnds` and `y_bnds`
      defines the local shelf orientation used for rotation.
    """
    
    # Import libraries
    import numpy as np
    import matplotlib.pyplot as plt 
    from pyproj import Proj, Transformer
    from datetime import timedelta, datetime

    #------------------------------------------# 
    # STEP #1 - Extract the isobath contour coordinates
    #------------------------------------------# 

    # Use contouring to extract the specified isobath(s)
    contour_set = plt.contour(lon_grid, lat_grid, bathy_grid, levels=[isobath])
    plt.close()  # Prevents plot from displaying if you're in a notebook

    # Extract all contour paths at the depth level
    paths = contour_set.allsegs[0] 

    # Filter based on number of points (to exclude small closed loops like seamounts)
    min_points = 100  # Adjust based on your resolution
    valid_paths = [p for p in paths if len(p) > min_points]

    # Select the longest or first valid path
    if isobath >= -500:
        selected_path = valid_paths[1]  # Or choose based on location
    else: 
        selected_path = valid_paths[0]  # Or choose based on location

    # Extract lon/lat from the path
    lon_iso = selected_path[:, 0]
    lat_iso = selected_path[:, 1]

    # Plot the isobath contours and selected path
    if option_plot == True:

        # Loop through paths
        for i, p in enumerate(paths):

            # Plot the ith contour path
            plt.plot(p[:, 0], p[:, 1], '.', label=f'Path {i}')

        # Plot the selected path 
        plt.plot(lon_iso,lat_iso, 'k-', label='Selected Path')

        # Set figure attributes
        plt.axis('equal')
        plt.grid(True,linestyle='--')
        plt.legend()
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.title("Extracted" + str(isobath) + "Isobath Paths")
        plt.show()

    #------------------------------------------# 
    # STEP #2 - Convert the isobath coordinates to easting and northing coordinates
    #------------------------------------------# 

    # Easting and Northing projection for the region, e.g., UTM zone 10N for California
    transformer = Transformer.from_crs("epsg:4326", "epsg:32610", always_xy=True)  # WGS84 to UTM Zone 10N

    # Convert lon/lat to x/y in meters
    x_iso, y_iso = transformer.transform(lon_iso, lat_iso)

    # Adjust isobath coordinates
    x_iso_n, y_iso_n = (1/1000)* (x_iso - np.mean(x_iso)), (1/1000)* (y_iso - np.mean(y_iso)) 

    # Plot the smoothed isobath in easting and northing coordinates
    if option_plot == True:

        # Plot the unsmoothed and smoothed isobaths 
        plt.plot(x_iso_n,y_iso_n, 'k-', label= str(isobath) + ' meter Isobath')

        # Set figure attributes
        plt.axis('equal')
        plt.grid(True,linestyle='--')
        plt.legend()
        plt.xlabel("Easting")
        plt.ylabel("Northing")
        plt.show()

    #------------------------------------------# 
    # STEP #3 - Calculate the Orientation Angle for the isobath 
    #------------------------------------------# 

    # Mask easting and northing vectors based on limits above
    x_mask = (x_iso_n >= x_bnds[0]) & (x_iso_n <= x_bnds[1]) 
    y_mask = (y_iso_n >= y_bnds[0]) & (y_iso_n <= y_bnds[1]) 
    mask = x_mask & y_mask 

    # Apply mask
    x_iso_mask = x_iso_n[mask]
    y_iso_mask = y_iso_n[mask]

    # Fit a straight line to the bathymetry
    m, b = np.polyfit(x_iso_mask, y_iso_mask, 1)  

    # Compute the angle
    theta = np.arctan(m)  

    # Orient the angle so positive along-shelf is upcoast (northernly) and positive cross-shelf is offshore (westerly) 
    theta_n = theta - (np.pi)

    # Plot the linear fit and print the rotation angle 
    if option_plot == True:

        # Print the angle in units of degrees
        print(np.rad2deg(theta) % 360)
        print(np.rad2deg(theta_n) % 360)

        # Compute the linear fit line 
        bfit = m*(x_iso_mask) + b

        # Plot the isobath and linear fit 
        plt.plot(x_iso_n,y_iso_n, 'k-', label= str(isobath) + ' Isobath')
        plt.plot(x_iso_mask,y_iso_mask, 'r-', label = ' Fitting section of Isobath')
        plt.plot(x_iso_mask,bfit, 'b-', label='Linear LSF')

        # Set figure attributes
        plt.axis('equal')
        plt.grid(True,linestyle='--')
        plt.legend()
        plt.xlabel("Easting (km)")
        plt.ylabel("Northing (km)")
        plt.show()

    #------------------------------------------# 
    # STEP #4 - Compute the depth averaged velocity (ALL distances)
    #------------------------------------------# 

    # Take absolute value of depth
    depth_pos = np.abs(depth)

    # Ensure increasing order (required by np.trapz)
    if not np.all(np.diff(depth_pos) > 0):
        depth_pos = depth_pos[::-1]
        u = u[:, :, ::-1]
        v = v[:, :, ::-1]

    # Mask depth levels below threshold if requested
    if option_mask == 1:

        # Mask depth levels deeper than depth_thresh
        mask_depth = depth_pos <= depth_thresh

        # Select shallower depths
        depth_sel = depth_pos[mask_depth]
        u_sel = u[:, :, mask_depth]
        v_sel = v[:, :, mask_depth]

        # Depth range
        H = depth_sel[-1] - depth_sel[0]

        # Compute weighted average
        u_bar_tmp = np.trapz(u_sel.filled(np.nan), depth_sel, axis=2) / H
        v_bar_tmp = np.trapz(v_sel.filled(np.nan), depth_sel, axis=2) / H

    else:

        # Depth range
        H = depth_pos[-1] - depth_pos[0]

        # Compute weighted average
        u_bar_tmp = np.trapz(u.filled(np.nan), depth_pos, axis=2) / H
        v_bar_tmp = np.trapz(v.filled(np.nan), depth_pos, axis=2) / H

    # Convert back to masked arrays
    u_bar = np.ma.masked_invalid(u_bar_tmp)
    v_bar = np.ma.masked_invalid(v_bar_tmp)

    #------------------------------------------# 
    # STEP #5 - Rotate velocity vectors (ALL distances)
    #------------------------------------------# 

    # Compute rotation matrix components
    cos_a = np.cos(theta_n)
    sin_a = np.sin(theta_n)

    # --- Depth-dependent velocities ---#
    u_along_tmp = u * cos_a + v * sin_a
    v_cross_tmp = -u * sin_a + v * cos_a

    # Mask zeros
    u_along = np.ma.masked_where(u_along_tmp == 0, u_along_tmp)
    v_cross = np.ma.masked_where(v_cross_tmp == 0, v_cross_tmp)

    # --- Depth-averaged velocities ---#
    u_along_bar = u_bar * cos_a + v_bar * sin_a
    v_cross_bar = -u_bar * sin_a + v_bar * cos_a

    return u_bar, v_bar,  u_along, v_cross, u_along_bar, v_cross_bar, theta_n
    