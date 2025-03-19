# Plotting functions for the scaling analysis
## Luke Colosi | lcolosi@ucsd.edu 

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