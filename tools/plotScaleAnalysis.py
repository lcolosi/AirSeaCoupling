# Plotting functions for the scaling analysis
## Luke Colosi | lcolosi@ucsd.edu 

#-- Figure corner labeling ---#
def add_corner_label(ax, pos, label, fontsize=12):
    """
    Add a labeled text box to a specified corner of an axes.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes object on which to place the label.
    pos : tuple of float
        The (x, y) location in axes coordinates where the label will be placed.
        Both values should be between 0 and 1, where (0, 0) is the lower-left
        corner and (1, 1) is the upper-right corner of the axes.
    label : str
        The text content of the label.
    fontsize : int, optional
        Font size of the label text. Default is 12.

    Returns
    -------
    None
        The function adds the label directly to the axes.
    """

    # Place text in lower left corner inside the axes
    ax.text(
        pos[0], pos[1], label,
        transform=ax.transAxes,
        fontsize=fontsize,
        fontweight='normal',
        va='center', ha='center',
        bbox=dict(
            boxstyle='square,pad=0.3',
            facecolor=(1, 1, 1, 0.6),  # faded white fill with alpha=0.6
            edgecolor='black',
            linewidth=1
        )
    )


#--- Time axis labeling ---# 
def month_fmt(x, pos):
    """
    Custom formatter function for labeling months on a Matplotlib time axis.

    Parameters
    ----------
    x : float
        The x-axis value representing time in Matplotlib's internal date format 
        (i.e., days since 0001-01-01 UTC, plus fractions of a day).
    pos : int
        The tick position index (required by Matplotlib's formatter interface, 
        but not used in this function).

    Returns
    -------
    label : str
        A formatted string for the x-axis tick label — either the first letter of 
        the month or, for January, the letter 'J' followed by the year on a new line.
    """

    # Import libraries 
    import matplotlib.dates as mdates

    # Convert the numeric x-axis value into a datetime object
    dt = mdates.num2date(x)

    # If the month is January, return 'J' with the year printed below (newline)
    # This helps visually mark the start of each year in the time series
    if dt.month == 1:
        return f"J\n{dt.year}"  # e.g., "J\n2023"
    
    # For all other months, return only the first letter of the abbreviated month name
    # Example: 'F' for February, 'M' for March, etc.
    else:
        return dt.strftime('%b')[0]
    
#--- Plotting regional decorrelation scale maps function ---# 
def plot_regional_decor(ax,data,lon, lat, lon_grid,lat_grid,bathy_grid,projection,resolution,lon_min,lon_max,lat_min,
                        lat_max,cmap,levels,xticks,yticks,fontsize,vmin,vmax,lon1,lat1,lon2,lat2,lon3,lat3,norm,Lt1,Lt2,
                        option_cce,option_cce_label): 

    """
    Function for plotting a regional decorrelation-scale map with bathymetry
    contours and optional CCE mooring markers.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis on which to plot the map.
    data : array_like
        2D decorrelation scale field on the model or observational grid
        (dimensions must match lon_grid and lat_grid).
    lon_grid : array_like
        2D longitude grid for pcolormesh/contourf (same shape as data).
    lat_grid : array_like
        2D latitude grid for pcolormesh/contourf (same shape as data).
    bathy_grid : array_like
        2D bathymetry field (negative values for ocean depth), same shape as lon_grid.
    projection : cartopy.crs.Projection
        Cartopy projection used for the map axes (e.g., ccrs.Mercator()).
    resolution : str
        Resolution string passed to cartopy_figs.set_subplots (e.g., '50m', '110m').
    lon_min, lon_max : float
        Longitude bounds of the plotted region.
    lat_min, lat_max : float
        Latitude bounds of the plotted region.
    cmap : matplotlib.colors.Colormap
        Colormap used for the decorrelation field.
    levels : array_like
        Contour levels for the filled decorrelation map.
    xticks, yticks : array_like
        Longitude and latitude tick locations for the grid lines.
    fontsize : int
        Base font size for labels and tick labels.
    vmin, vmax : float
        Minimum and maximum values for color normalization of the decorrelation field.
    lon1, lat1 : float
        Longitude and latitude of the first mooring (CCE1).
    lon2, lat2 : float
        Longitude and latitude of the second mooring (CCE2).
    lon3, lat3 : float
        Longitude and latitude of the third mooring (CCE3).
    norm : matplotlib.colors.Normalize
        Normalization object used to map Lt values to colormap space.
    Lt1, Lt2 : float
        Decorrelation scales at the first and second moorings (used for marker colors).
    option_cce : bool
        If True, plot the CCE mooring locations on the map.
    option_cce_label : bool or int
        If True (or 1), add text labels ("CCE1", "CCE2") next to the mooring markers.

    Returns
    -------
    cf : matplotlib.contour.QuadContourSet
        Filled contour object for the decorrelation map (useful for colorbars).
    """

    # Import helper functions for setting up cartopy axes and grid
    import cartopy_figs as cart
    import cmocean
    import cartopy.crs as ccrs
    import matplotlib.pyplot as plt
    import numpy as np

    # Configure the cartopy subplot (extent, coastlines, land, etc.)
    cart.set_subplots(ax, projection, resolution, lon_min=lon_min, lon_max=lon_max, lat_min=lat_min, lat_max=lat_max) 

    # Plot decorrelation scale
    cf = ax.contourf(
                    lon, lat, data, levels=levels, 
                    transform=ccrs.PlateCarree(),
                    cmap=cmap,  
                    shading="auto", 
                    extend='both',
                    vmin=vmin, vmax=vmax
                    )

    # Optionally plot CCE moorings and labels
    if option_cce == True: 

        #--- CCE1 ---# 

        # Plot the first mooring marker; color encodes decorrelation scale Lt1
        ax.scatter(
            lon1, lat1, 
            color = 'w', #cmocean.cm.amp(norm(Lt1)),  # color from the same colormap
            edgecolor='black', marker='^', s=40,  # customize as needed
            linewidth=0.5,
            transform=ccrs.PlateCarree(),
            zorder=10,
            label='CCE1'
        )

        
        if option_cce_label == 1:

            # Add a label next to the mooring marker
            ax.text(
                lon1 - 0.065, lat1 + 0.08,           # Slight offset to avoid overlapping the point
                f"CCE1",                # Text label showing decorrelation scale in months
                transform=ccrs.PlateCarree(),
                fontsize=fontsize-5,
                fontweight='bold',
                color='black',
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=1.5, alpha=0.5),
                zorder=11
            )

        #--- CCE2 ---# 

        # Plot the second mooring marker; color encodes decorrelation scale Lt2
        ax.scatter(
            lon2, lat2, 
            color = 'w', #cmocean.cm.amp(norm(Lt2)),  # color from the same colormap
            edgecolor='black', marker='s', s=40,  # customize as needed
            linewidth=0.5,
            transform=ccrs.PlateCarree(),
            zorder=10,
            label='CCE2'
        )

        if option_cce_label == True:

            # Add a label next to the mooring marker
            ax.text(
                lon2 - 0.065, lat2 + 0.08,           # Slight offset to avoid overlapping the point
                f"CCE2",                
                transform=ccrs.PlateCarree(),
                fontsize=fontsize-5,
                fontweight='bold',
                color='black',
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=1.5, alpha=0.5),
                zorder=11
            )

        #--- CCE3 ---# 

        # Plot the first mooring marker; color encodes decorrelation scale Lt1
        ax.scatter(
            lon3, lat3, 
            color = 'w', #cmocean.cm.amp(norm(Lt1)),  # color from the same colormap
            edgecolor='black', marker='o', s=40,  # customize as needed
            linewidth=0.5,
            transform=ccrs.PlateCarree(),
            zorder=10,
            label='CCE3'
        )
    
    # --- Bathymetry contours --- #

    # Define contour levels for shelf and deeper bathymetry (positive depths)
    level_innershelf = np.arange(0, 300, 100)
    levels_midshelf = np.arange(1000, 3000, 500)

    # Plot deeper isobaths (e.g., 1000–2500 m, dashed)
    contour1 = ax.contour(
        lon_grid, lat_grid, -1 * bathy_grid,
        levels=levels_midshelf,
        colors="black",
        linewidths=0.5,
        linestyles="dashed",
    )

    # Highlight the 2000 m isobath with a solid line
    contour2 = ax.contour(
        lon_grid, lat_grid, -1 * bathy_grid,
        levels=[2000],
        colors="black",
        linewidths=1,
        linestyles="solid",
    )

    # Plot inner-shelf isobaths (e.g., 0–200 m, dashed)
    contour3 = ax.contour(
        lon_grid, lat_grid, -1 * bathy_grid,
        levels=level_innershelf,
        colors="black",
        linewidths=0.5,
        linestyles="dashed",
    )

    # Highlight the 200 m isobath with a solid line
    contour4 = ax.contour(
        lon_grid, lat_grid, -1 * bathy_grid,
        levels=[200],
        colors="black",
        linewidths=1,
        linestyles="solid",
    )

    # Label the depth contours
    plt.clabel(contour1, fontsize=6)
    plt.clabel(contour2, fontsize=6)
    plt.clabel(contour3, fontsize=6)
    plt.clabel(contour4, fontsize=6)

    # --- Grid and tick labels --- #

    # Add lat/lon grid lines and axis tick labels
    cart.set_grid_ticks(
        ax,
        projection=ccrs.PlateCarree(),
        xticks=xticks,
        yticks=yticks,
        xlabels=True,
        ylabels=True,
        grid=True,
        fontsize=fontsize,
        color="black",
    )
    
    return cf

#--- Plotting Variance Ellipses in Polar coordinates ---# 
def plot_decorrelation_ellipses_polar(ellipses, use_inverse=False, fill=False,
                                      show_axes=True, show_legend=True, ax=None,
                                      alpha=0.3, units="km", rmax=None):
    """
    Plot one or more decorrelation-scale ellipses in polar coordinates
    (θ measured clockwise from North, r = decorrelation scale).

    Parameters
    ----------
    ellipses : list of dict
        Each dict must contain:
        {
            'a': float,          # major decorrelation scale
            'b': float,          # minor decorrelation scale
            'theta_deg': float,  # orientation clockwise from North (degrees)
            'label': str,        # label for legend
            'color': str         # color string for plotting
        }
    use_inverse : bool, default=False
        If True, plot the *inverse decorrelation scale* (1/a, 1/b),
        useful for visualizing wavenumber-like quantities.
    fill : bool, default=False
        If True, fill the ellipse with semi-transparent color.
    show_axes : bool, default=True
        If True, plot dashed major and dotted minor axes.
    show_legend : bool, default=True
        If True, display legend.
    ax : matplotlib.axes._subplots.PolarAxesSubplot, optional
        Existing polar axis to plot on. If None, a new one is created.
    alpha : float, default=0.3
        Transparency for filled ellipses (only used if fill=True).
    units : str, default="km"
        Units for radial axis label. If `use_inverse=True`, the units
        will be shown as reciprocal (e.g., 1/km).
    rmax : float, optional
        Maximum value for the radial axis (upper limit). If None, it is set
        automatically to 10% above the largest ellipse radius.

    Returns
    -------
    ax : PolarAxesSubplot
        The matplotlib polar axes with all ellipses plotted.
    """

    import numpy as np
    import matplotlib.pyplot as plt

    # Create polar axes if none provided
    if ax is None:
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(6, 6))

    # Track maximum radius for autoscaling
    max_r = 0.0

    # Loop through ellipses
    for e in ellipses:
        a = e['a']
        b = e['b']
        theta_deg = e['theta_deg']
        color = e.get('color', 'r')
        label = e.get('label', None)

        # Optionally invert the scales (e.g., to visualize wavenumber)
        if use_inverse:
            a, b = 1.0 / a, 1.0 / b

        # Update maximum radius tracker
        max_r = max(max_r, a, b)

        # Build ellipse in local coordinates
        t = np.linspace(0, 2 * np.pi, 600)
        x_loc = a * np.cos(t)
        y_loc = b * np.sin(t)

        # Convert orientation (clockwise-from-North → CCW-from-East)
        phi_deg = 90.0 - theta_deg
        phi = np.deg2rad(phi_deg)

        # Rotate into (x=East, y=North)
        R = np.array([[np.cos(phi), -np.sin(phi)],
                      [np.sin(phi),  np.cos(phi)]])
        x, y = R @ np.vstack((x_loc, y_loc))

        # Convert to polar coordinates (north=0°, clockwise)
        r = np.hypot(x, y)
        theta_polar = (np.pi / 2 - np.arctan2(y, x)) % (2 * np.pi)

        # Plot ellipse (filled or outline)
        if fill:
            ax.fill(theta_polar, r, color=color, alpha=alpha, label=label)
        else:
            ax.plot(theta_polar, r, lw=2, color=color, label=label)

        # Plot major/minor axes (optional)
        if show_axes:
            th_major = np.deg2rad(theta_deg)
            th_minor = (th_major + np.pi / 2) % (2 * np.pi)
            ax.plot([th_major, th_major + np.pi], [0, a], '--', color=color, lw=1)
            ax.plot([th_minor, th_minor + np.pi], [0, b], ':', color=color, lw=1)

    # --- Set radial axis limits ---
    if rmax is None:
        rmax = max_r * 1.1  # add 10% padding
    ax.set_rlim(0, rmax)

    # --- Polar formatting ---
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_rlabel_position(135)

    # --- Append units to the last radial tick label (inline) ---
    def add_units_to_last_radial_tick(ax, units=units, use_inverse=use_inverse, fmt="{:.1f}"):
        """
        Append units to the last radial tick label on a polar axis.
        """
        ax.figure.canvas.draw()
        r_ticks = np.asarray(ax.get_yticks())
        if r_ticks.size == 0:
            r_ticks = np.linspace(rmax / 6, rmax, 6)

        labels = [fmt.format(t) for t in r_ticks]
        unit_label = f"(1/{units})" if use_inverse else f"({units})"
        labels[-1] = f"{labels[-1]} {unit_label}"

        angle = ax.get_rlabel_position()
        ax.set_rgrids(r_ticks, labels, angle=angle)

    add_units_to_last_radial_tick(ax, units=units, fmt="{:.1f}")

    # Add legend (optional)
    if show_legend:
        ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1))

    return ax




#--- Autocorrelation and Autocovariance Function ---# 
def plot_depth_data_autocorr(dist, dist_on, dist_trans, dist_off, water_depth, water_depth_on, water_depth_trans, water_depth_off, data, data_on, data_trans, data_off, autocorr_on, autocorr_trans, autocorr_off, on_lim, off_lim, fontsize, dirOut):

    """
    plot_depth_data_autocorr(dist, dist_on, dist_trans, dist_off, water_depth, water_depth_on, water_depth_trans, water_depth_off, data, data_on, data_trans, data_off, autocorr_on, autocorr_trans, autocorr_off, on_lim, off_lim, fontsize, dirOut):

    Function for plotting the water depth, data transects along side the autocorrelation function for each glider transect. 
    
        Parameters
        ----------
        data : array
            
        Returns
        -------

        Libraries necessary to run function
        -----------------------------------
        import numpy as np 
        import matplotlib.pyplot as plt

    """

    # Import libraries 
    import numpy as np
    import matplotlib.pyplot as plt

    # Obtain the distance from shore of the on and off-shelf depth criteria 
    idx_on = np.abs(water_depth - on_lim).argmin()
    idx_off = np.abs(water_depth - off_lim).argmin()
    dist_on_lim = dist[idx_on]
    dist_off_lim = dist[idx_off]

    # Set plotting parameters
    plt.rcParams.update({'font.size': fontsize})  
    plt.rcParams['text.usetex'] = False

    # Create figure
    fig, axes = plt.subplots(3, 1, figsize=(10, 8))

    #--- Suplot #1 ---# 

    # Plot water depth 
    axes[0].plot(dist,water_depth,'k.-')
    axes[0].plot(dist_on,water_depth_on,'b.-', label=r'On-shelf: $\geq$' + str(on_lim) + ' m')
    axes[0].plot(dist_off,water_depth_off,'g.-', label=r'Off-shelf: $\leq$' + str(off_lim) + ' m')
    axes[0].plot(dist_trans,water_depth_trans,'r.-', label=r'Shelf-break')

    # Plot on and off-shelf limits
    axes[0].axvline(dist_on_lim, color='b', linestyle='--', linewidth=2)
    axes[0].axvline(dist_off_lim, color='g', linestyle='--', linewidth=2)

    # Add label text near the vertical lines
    #axes[0].text(dist_on_lim - 1,np.min(water_depth)+150, r'On-shelf: $\geq$' + str(on_lim) + ' m', color='b', fontsize=12, ha='left',va='top',rotation=0)
    #axes[0].text(dist_off_lim + 1,np.max(water_depth)+150, r'Off-shelf: $\geq$' + str(off_lim) + ' m', color='g', fontsize=12, ha='right',va='top',rotation=0)

    # Set axis attributes
    axes[0].set_ylabel(r"Water Depth (m)")
    axes[0].set_xlim(0,400)
    #axes[0].set_xlim(0,50)
    axes[0].set_ylim(-5000,0)
    axes[0].grid()

    # Reverse the direction of the x-axis 
    axes[0].invert_xaxis()

    #--- Suplot #2 ---# 

    # Plot data
    axes[1].plot(dist,data,'k.-')
    axes[1].plot(dist_on,data_on,'b.-')
    axes[1].plot(dist_off,data_off,'g.-')
    axes[1].plot(dist_trans,data_trans,'r.-')

    # Plot on and off-shelf limits
    axes[1].axvline(dist_on_lim, color='b', linestyle='--', linewidth=2)
    axes[1].axvline(dist_off_lim, color='g', linestyle='--', linewidth=2)

    # Set axis attributes
    axes[1].set_xlabel(r"Distance from Shore (km)")
    axes[1].set_ylabel(r"Temperature ($^\circ$C)")
    axes[1].set_xlim(0,400)
    axes[1].grid()

    # Reverse the direction of the x-axis 
    axes[1].invert_xaxis()

    #--- Suplot #3 ---# 

    # Plot autocorrelation function 
    axes[2].plot(dist_on - dist_on[0],autocorr_on,'b.-')
    axes[2].plot(dist_trans - dist_trans[0],autocorr_trans,'r.-')
    axes[2].plot(dist_off - dist_off[0],autocorr_off,'g.-')

    # Set axis attributes
    axes[2].set_xlabel(r"Distance Scale $\delta$ (km)")
    axes[2].set_ylabel(r"$R_{TT}(\delta)$")
    #axes[2].set_xlim(0,300)
    axes[2].grid()

    # Reverse the direction of the x-axis 
    axes[2].invert_xaxis()

    # Display figure
    plt.tight_layout()
    plt.show()

    # Save figure 
    fig.savefig(fname = dirOut, bbox_inches = 'tight', dpi=300)

    return


#--- Plot frequency markers ---# 
def add_freq_marker(ax, freq, label,
                    y_marker=1.01,
                    y_text=1.07,
                    ms=6,
                    x_text_offset_pts=0.0,
                    markerfacecolor='white',
                    markeredgecolor='k',
                    fontsize=10):
    """
    Add an upside-down triangular frequency marker just above the top
    of an axes, with a text label above it.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes on which to draw the marker and label.
    freq : float
        Frequency location in data units (e.g., cycles/day) where the
        marker should be placed along the x-axis.
    label : str
        Text label to place above the marker (e.g., 'K1', 'f', 'M2').
    y_marker : float, optional
        Vertical position of the marker in axes coordinates (0 = bottom,
        1 = top). Values slightly > 1 place the marker just above the
        top of the plotting area. Default is 1.01.
    y_text : float, optional
        Vertical position of the text label in axes coordinates. Default
        is 1.07.
    ms : float, optional
        Marker size. Default is 6.
    x_text_offset_pts : float, optional
        Horizontal offset of the text label relative to the marker, in
        points. Positive values shift the label to the right; negative
        values shift it to the left. Default is 0.0.
    markerfacecolor : str or tuple, optional
        Face color of the triangular marker. Default is 'white'.
    markeredgecolor : str or tuple, optional
        Edge color of the triangular marker. Default is 'k'.
    fontsize : float, optional
        Fontsize of the text label. Default is 10. 

    Returns
    -------
    None
    """

    # Import libraries 
    import matplotlib.transforms as transforms

    # Blended transform: x in data coordinates, y in axes coordinates.
    # This keeps the marker at the correct frequency while placing it
    # just above the top of the panel.
    base_trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)

    # Plot upside-down triangular marker with an outline so it stands out
    # against the spectrogram.
    ax.plot(
        freq, y_marker,
        marker='v',
        linestyle='None',
        markersize=ms,
        markerfacecolor=markerfacecolor,
        markeredgecolor=markeredgecolor,
        transform=base_trans,
        clip_on=False,   # allow drawing outside the axes region
        zorder=10
    )

    # Create a new transform that is shifted horizontally by
    # x_text_offset_pts in display (point) units. This makes the label
    # easy to nudge left/right independently of the log-scaled x-axis.
    text_trans = transforms.offset_copy(
        base_trans, fig=ax.figure,
        x=x_text_offset_pts, y=0.0, units='points'
    )

    # Add the text label above the marker.
    ax.text(
        freq, y_text, label,
        ha='center', va='bottom',
        transform=text_trans,
        clip_on=False,
        zorder=10, 
        fontsize=fontsize
    )
    return

#--- Scale bar for Cartopy maps ---#
def add_scalebar(ax, length_km=50, location=(0.15, 0.07),
                 linewidth=2, text_kwargs=None):
    """
    Add a horizontal scale bar to a Cartopy GeoAxes.

    Parameters
    ----------
    ax : cartopy.mpl.geoaxes.GeoAxes
        The axes to draw the scale bar on.
    length_km : float
        Length of the scale bar in kilometers.
    location : tuple of float
        Location of the center of the scale bar in *axes* coordinates
        (x, y), both between 0 and 1.
    linewidth : float
        Line width of the scale bar.
    text_kwargs : dict, optional
        Extra kwargs passed to ax.text (fontsize, weight, etc.).
    """

    # Import libraries
    import numpy as np
    import cartopy.crs as ccrs
    import matplotlib.pyplot as plt

    if text_kwargs is None:
        text_kwargs = dict(fontsize=10)

    # Get current map extent in data (PlateCarree) coordinates
    lon_min, lon_max, lat_min, lat_max = ax.get_extent(crs=ccrs.PlateCarree())

    # Choose the latitude where the bar will be drawn
    lat = lat_min + location[1] * (lat_max - lat_min)

    # Convert desired length (km) to degrees of longitude at that latitude
    km_per_deg_lon = 111.32 * np.cos(np.deg2rad(lat))
    dlon = (length_km / km_per_deg_lon)

    # Choose center longitude based on axes location
    lon_center = lon_min + location[0] * (lon_max - lon_min)
    lon_left   = lon_center - dlon / 2
    lon_right  = lon_center + dlon / 2

    # Small vertical tick height in degrees
    dtick = 0.01 * (lat_max - lat_min)

    # Horizontal bar
    ax.plot([lon_left, lon_right], [lat, lat],
            transform=ccrs.PlateCarree(),
            color='k', linewidth=linewidth, solid_capstyle='butt')

    # Vertical ticks at ends (to make |-----|)
    ax.plot([lon_left, lon_left],
            [lat - dtick, lat + dtick],
            transform=ccrs.PlateCarree(),
            color='k', linewidth=linewidth)
    ax.plot([lon_right, lon_right],
            [lat - dtick, lat + dtick],
            transform=ccrs.PlateCarree(),
            color='k', linewidth=linewidth)

    # Label above the bar
    ax.text(lon_center, lat + 1.8 * dtick,
            f"{int(length_km)} km",
            ha='center', va='bottom',
            transform=ccrs.PlateCarree(),
            **text_kwargs)
    return

def add_longitude_axis(
    ax,
    dist,
    lon,
    lon_markers=None,
    marker_colors=None,
    label_every=2,
    lon_label='',
    add_vlines=True,
    vline_kwargs=None,
    marker_func=None,
    marker_kwargs=None,
):
    """
    Add a top longitude axis to a distance-based plot.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Main axis.

    dist : ndarray
        Distance coordinate.

    lon : ndarray
        Longitude values corresponding to dist.

    lon_markers : list or ndarray, optional
        Longitudes where markers/vertical lines should be added.

    marker_colors : list, optional
        Colors for each longitude marker.

    label_every : int, optional
        Keep every nth longitude label.

    lon_label : str, optional
        Label for top axis.

    add_vlines : bool, optional
        Whether to draw vertical lines on main axis.

    vline_kwargs : dict, optional
        Keyword arguments passed to ax.axvline().

    marker_func : callable, optional
        Function used to add markers on top axis.

    marker_kwargs : dict, optional
        Keyword arguments passed to marker_func.

    Returns
    -------
    ax_top : matplotlib.axes.Axes
        Top longitude axis.

    marker_distances : ndarray
        Distances corresponding to lon_markers.
    """

    # Import libraries
    import numpy as np

    # --- Top longitude axis --- #
    ax_top = ax.twiny()
    ax_top.set_xlim(ax.get_xlim())
    ax_top.set_xlabel(lon_label)

    # Convert longitude to [-180, 180]
    lon_180 = ((lon + 180) % 360) - 180

    # Tick positions
    dist_ticks = ax.get_xticks()

    # Interpolate longitude onto tick positions
    lon_ticks = np.interp(dist_ticks, dist, lon_180)

    # Create tick labels
    labels = [
        f"{abs(x):.1f}°W" if i % label_every == 0 else ""
        for i, x in enumerate(lon_ticks)
    ]

    ax_top.set_xticks(dist_ticks)
    ax_top.set_xticklabels(labels)

    # --- Marker interpolation prep --- #
    sort_idx = np.argsort(lon_180)
    lon_sorted = lon_180[sort_idx]
    dist_sorted = dist[sort_idx]

    marker_distances = None

    # --- Add markers and vertical lines --- #
    if lon_markers is not None:

        lon_markers = np.atleast_1d(lon_markers)

        marker_distances = np.interp(
            lon_markers,
            lon_sorted,
            dist_sorted
        )

        if marker_colors is None:
            marker_colors = ['k'] * len(lon_markers)

        if vline_kwargs is None:
            vline_kwargs = {}

        if marker_kwargs is None:
            marker_kwargs = {}

        for d, c in zip(marker_distances, marker_colors):

            # Add top marker
            if marker_func is not None:
                marker_func(
                    ax_top,
                    d,
                    '',
                    markerfacecolor=c,
                    markeredgecolor=c,
                    **marker_kwargs
                )

            # Add vertical line
            if add_vlines:
                ax.axvline(
                    d,
                    color=c,
                    linestyle='--',
                    lw=1.5,
                    alpha=0.7,
                    **vline_kwargs
                )

    return ax_top, marker_distances