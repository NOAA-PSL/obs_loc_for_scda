import netCDF4 as nc
import xarray as xr
#from cartopy import config
#import cartopy.crs as ccrs
#import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
#import re
#from scipy import stats
from load_data_fns import *

## Where are we working
proj_dir = '/work/noaa/gsienkf/zstanley/projects/obs_loc'
data_dir = proj_dir + '/data'
plot_dir = proj_dir + '/plots'

## Open netcdf files with xarray
ds = xr.open_mfdataset(data_dir+'/ens1*.nc', autoclose=True, preprocess=preprocess)

## Get means and variances
ds_mean = ds.mean('ens_mem')

## Select 25 columns in the Indian Ocean
lon_sel_pt = 80
lat_sel_pt = -25
lon_sel = slice(lon_sel_pt-2, lon_sel_pt+3, 1)
lat_sel = slice(lat_sel_pt-2, lat_sel_pt+3, 1)
ds_sel = ds_mean['ocn_Temp'].sel(lon=lon_sel, lat=lat_sel)

## Plot columns
ds_np = ds_sel.to_numpy()
ds_np = ds_np.reshape((75, 25))
plt.plot(ds_np, ds_sel['ocn_lev'])
plt.gca().invert_yaxis()
plt.title('Temperature profiles in 25 columns in Indian Ocean')
plt.xlabel('Temperature (C)')
plt.ylabel('Depth (m)')
plt.savefig(plot_dir + '/indian_ocean_temperature_profiles')
plt.show()

## Remove the mean of the 25 column means and replot
ds_np_mean = np.mean(ds_np, 1)
ds_np_anom = ds_np - np.tile(ds_np_mean, (25, 1)).transpose()
plt.plot(ds_np_anom, ds_sel['ocn_lev'])
plt.gca().invert_yaxis()
plt.title('Temperature profiles, anomalies in 25 column means in Indian Ocean')
plt.xlabel('Temperature anomalies (C)')
plt.ylabel('Depth (m)')
plt.savefig(plot_dir + '/indian_ocean_temperature_profiles_anomalies_in_column_mean')
plt.show()

## Plot only the top 150 m
plt.plot(ds_np_anom[:37], ds_sel['ocn_lev'][:37])
plt.gca().invert_yaxis()
plt.title('Temperature profiles, anomalies in 25 column means in Indian Ocean')
plt.xlabel('Temperature anomalies (C)')
plt.ylabel('Depth (m)')
plt.savefig(plot_dir + '/indian_ocean_temperature_profiles_anomalies_in_column_mean_150m')
plt.show()

## Plot standard deviation of background error
ds_manual = np.empty((75, 2000))
start = 0
end = 80
for lon in range(lon_sel_pt-2, lon_sel_pt+3, 1):
  for lat in range(lat_sel_pt-2, lat_sel_pt+3, 1):
    # Select ocn columns
    ds_ocn = ds['ocn_Temp'].sel(lon=lon, lat=lat, method='nearest')
    ds_ocn_np = ds_ocn.to_numpy()
    ds_ocn_np = np.swapaxes(ds_ocn_np, 0, 1)
    ens_ocn_avg = ds_ocn_np.mean(axis=1)
    ens_ocn_avg_expand = np.tile(ens_ocn_avg, (80,1)).transpose()
    # Store ocn columns
    ds_manual = ds_ocn_np - ens_ocn_avg_expand
    start = start + 80
    end = end + 80

## Plot standard deviation of background error
cov_temp = np.cov(ds_manual)
spread = np.sqrt(np.diag(cov_temp))
plt.plot(spread[:37], ds['ocn_lev'][:37])
plt.gca().invert_yaxis()
plt.xlabel('Spread (std. dev.)')
plt.ylabel('Depth (m)')
plt.title('Spread in 2000 member bootstrapped ensemble, Indian Ocean')
plt.savefig(plot_dir+'/indian_ocean_2000_mem_ens_spread_150m.png')
plt.show()




