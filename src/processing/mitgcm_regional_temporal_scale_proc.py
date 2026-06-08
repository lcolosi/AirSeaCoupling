#---------------------------------------------------------------------# 
# MITGCM Regional Temporal Scale Analysis 

# Purpose: Code for preforming the regional time scale analysis. Here, 
# we compute the decorrelation time scale for a subset of the model 
# domain at each grid point. 
#---------------------------------------------------------------------# 

# Import python libraries
import sys
import xarray as xr
import numpy as np
from netCDF4 import Dataset, num2date
from datetime import datetime
import os
import glob

# Set path to access python functions
ROOT = '/Users/lukecolosi/Desktop/projects/graduate_research/Gille_lab/'
sys.path.append(ROOT + 'AirSeaCoupling/tools/')

# Import project functions
import cartopy_figs as cart
from autocorr import compute_autocorr_optimize, compute_decor_scale_optimize
from lsf import unweighted_lsf, detrend

#---------------------------------------------------------------------# 
# STEP 1 - Set data analysis parameters
#---------------------------------------------------------------------# 

# Set processing parameters
option_var           = 'vvel'  # Specifies which variable will analyzed. Options include: temp, sal, density, uvel, vvel

# Set time and space parameters
T            = 1*(60)*(60)                   # Spcifies the time interval for model run. Units: seconds
estimator    = 'biased'                      # Specifies the approach for estimating the autocorrelation function    
lat_bnds     = [33, 35]                      # Specifies the latitude bounds for the region to analyze
lon_bnds     = [-123 % 360, -120 % 360]      # Specifies the longitude bounds for the region to analyze

# Set path to project directory
PATH = ROOT + 'AirSeaCoupling/data/mitgcm/SWOT_MARA_RUN4_LY/spatial/'
PATH_bathy  = ROOT + 'AirSeaCoupling/data/bathymetry/'

#---------------------------------------------------------------------# 
# STEP 2 - Compute decorrelation scales at each depth 
#---------------------------------------------------------------------# 

# Extract file names
if (option_var == 'temp') | (option_var == 'sal') | (option_var == 'density'):
    filenames = glob.glob(PATH + "mitgcm_intermediate_data_TSD_hrly_map_depth_*m.nc")
elif (option_var == 'uvel') | (option_var == 'vvel'):
    filenames = glob.glob(PATH + "mitgcm_intermediate_data_vel_hrly_map_depth_*m.nc")

# Loop through files 
for f in filenames[1:]: 

    #---------------------------------------------------------------------# 
    # STEP 2A - Read in data variables 
    #---------------------------------------------------------------------# 

    # Open data set
    nc = Dataset(f, 'r')

    # Extract data variables
    depth = nc.variables['Depth'][:]
    lon   = nc.variables['lon'][:]
    lat   = nc.variables['lat'][:]
    time  =  num2date(nc.variables['time'][:], nc.variables['time'].units)

    if option_var == 'temp':
        data = nc.variables['CTemp'][:]
    elif option_var == 'sal':
        data = nc.variables['ASal'][:]
    elif option_var == 'density':
        data = nc.variables['SIG'][:]
    elif option_var == 'uvel':
        data = nc.variables['u'][:]
    elif option_var == 'vvel':
        data = nc.variables['v'][:]

    # Convert cftime.DatetimeGregorian to Python datetime objects
    time_dt = np.array([datetime(d.year, d.month, d.day, d.hour, d.minute, d.second) for d in time])

    # Convert to seconds since start time
    t0 = time_dt[0]
    time_elapsed = np.array([(t - t0).total_seconds() for t in time_dt])

    # Print depth 
    print('Processing ' + str(int(np.floor(abs(depth)))) + ' meter depth...')

    #---------------------------------------------------------------------# 
    # STEP 2B - Compute decorrelation scales
    #---------------------------------------------------------------------#
    
    # Set parameters
    ntime,nlat,nlon = np.shape(data)
    lag = ntime
    w1, w2 = [(2*np.pi)/31557600, (1/2)*(2*np.pi)/31557600]      # Radian Frequency for annual and semi-annual cycles. Units: rad/sec

    # Initialize arrays 
    autocorr       = np.zeros((ntime,nlat,nlon))
    time_scale     = np.zeros((ntime,nlat,nlon))
    Lt             = np.zeros((nlat,nlon))
    cn_lon = 0 

    # Loop through longitude 
    for ilon in range(0,nlon):
        
        # Set progress bar
        progress = (ilon + 1) / (len(lon)-1)
        sys.stdout.write(f"\rProgress: {progress:.1%}")
        sys.stdout.flush()

        # Set latitude counter 
        cn_lat = 0

        # Loop through latitude
        for ilat in range(0,nlat):
            
            # Initialize the ith time series 
            data_ts = data[:,ilat,ilon]

            # Try statement to handle masked values associated with land
            try:

                # Remove annual and semi-annual cycle 
                hfit, x_data, x_data_sigma, _ = unweighted_lsf(data_ts, time_elapsed, parameters = 2, freqs = np.array([w1,w2]), sigma = None)
                data_ts_rm = data_ts - hfit

                # Detrend data record 
                data_dt = detrend(data_ts_rm, time_elapsed, mean = 0)

                # Compute autocorrelation function
                autocorr[:,cn_lat,cn_lon], _, _, _, time_scale[:,cn_lat,cn_lon], _ = compute_autocorr_optimize(data_dt, time_elapsed, lag, estimator, 0)

                # Compute the decorrelation scale
                Lt[cn_lat,cn_lon] = compute_decor_scale_optimize(autocorr[:,cn_lat,cn_lon],time_scale[:,cn_lat,cn_lon],T,'unbiased',0)

            except Exception: 

                # Print warining message 
                print('Masked time series! Skipping grid point')

            # Set latitude counter
            cn_lat = cn_lat + 1

        # Set longitude counter
        cn_lon = cn_lon + 1

    # Convert time scale to units of days
    Lt_days = Lt*(1/60)*(1/60)*(1/24)

    #---------------------------------------------------------------------# 
    # STEP 3 - Save decorrelation scales into a npz file
    #---------------------------------------------------------------------#

    # Check if file exists, then delete it
    file_path = PATH + "/mitgcm_regional_temporal_scale_" + option_var  + "_depth_" + str(int(np.floor(abs(depth)))) + "m.npz"
    if os.path.exists(file_path):
        os.remove(file_path)

    # Set metadata
    metadata = {
        'description': 'Temporal decorrelation length scale from the MIT gcm model out of ' + option_var + ' at depth ' + str(np.round(depth,2)) + 'm. Here, the annual and semi-annual cycles are removed before computing the decorrelation scale.',
        'source': 'MITgcm model data from SWOT_MARA_RUN4_LY',
        'coordinates units': 'km'
    }

    # Save data arrays and metadata to a .npz file
    np.savez(file_path, 
            autocorr        = autocorr,
            Lt_days         = Lt_days,          # Units: days
            time_scale      = time_scale,       # Units: seconds
            lon             = lon,
            lat             = lat,
            depth           = depth,
            metadata        = metadata
            )
    

