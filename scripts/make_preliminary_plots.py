import netCDF4 as nc
import xarray as xr
from cartopy import config
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import re
from scipy import stats
from load_data_fns import *

## Where are we working
proj_dir = '/work/noaa/gsienkf/zstanley/projects/obs_loc'
data_dir = proj_dir + '/data'
plot_dir = proj_dir + '/plots'

## Open netcdf files with xarray
ds = xr.open_mfdataset(data_dir+'/ens1*.nc', autoclose=True, preprocess=preprocess)

## Get means and variances
ds_mean = ds.mean('ens_mem')
ds_std = ds.std('ens_mem')

## Get 2-d correlations
sst_t2m_corr = xr.corr(ds['sst'], ds['atm_t2m'], dim='ens_mem')
sst_windspd_corr = xr.corr(ds['sst'], ds['wind_spd'], dim='ens_mem')
sst_tprcp_corr = xr.corr(ds['sst'], ds['atm_tprcp'], dim='ens_mem')

## Get 3d correlations
sst_col_corr = xr.Dataset()
sst_col_corr['sst_atm_T'] = xr.corr(ds['sst'], ds['atm_T'], dim = 'ens_mem')
sst_col_corr['sst_ocn_T'] = xr.corr(ds['sst'], ds['ocn_Temp'], dim='ens_mem')
sst_col_corr['wind_spd_atm_T'] = xr.corr(ds['wind_spd'], ds['atm_T'], dim = 'ens_mem')
sst_col_corr['wind_spd_ocn_T'] = xr.corr(ds['wind_spd'], ds['ocn_Temp'], dim = 'ens_mem')
# Atm. layer height 
sst_col_corr['atm_z'] = ds_mean['atm_DZ'].cumsum(dim='atm_lev') - ds_mean['atm_DZ'].cumsum(dim='atm_lev').min(dim='atm_lev')
sst_col_corr.atm_z.attrs = {
  'long_name': 'Atmospheric geopotential height. Set to 0 at surface level',
  'units': 'Geo-potential meters'}
sst_col_corr['ocn_z'] = -1 * ds['ocn_lev'] 

## Plot single field
this_var = sst_t2m_corr
vmin=0.2
vmax=0.6
cmap = 'Purples' #'bwr'
plt_title = 'SST / AST'
ax = plt.axes(projection=ccrs.Mollweide())
im = ax.pcolormesh(ds['lon'], ds['lat'], this_var, vmin=vmin, vmax=vmax, cmap=cmap, transform=ccrs.PlateCarree())
ax.coastlines()
ax.gridlines(draw_labels=True)
plt.title(plt_title)
plt.colorbar(im)
plt.show()

## Plot vertical correlations
# Where are the columns
ax = plt.axes(projection=ccrs.PlateCarree())
plt.title('Column locations')
ax.set_extent([-180, 180, -80, 80], ccrs.PlateCarree())
ax.coastlines(resolution='110m')
plt.plot([160,-130, -90, 60], [40, 0, -30, -20],'ro',  markersize=5)
plt.savefig(plot_dir+'/corr_column_locations.png')

# Single column
plot_four_single_col(sst_col_corr, atm_var='wind_spd_atm_T', ocn_var='wind_spd_ocn_T')
plt.suptitle('Correlation between surface wind speed and temperature in select columns')
plt.tight_layout()
# Save
figdata = {'Title': 'Correlation between surface wind speed and temperature in select columns', 
           'Author': 'Zofia C Stanley',
           'Description': 'Correlations calculated in north pacific (160E,40N), tropical pacific (130W,0N), south pacific (90W,30S), and indian ocean (60E,20S).',
           'Creation Time': 'Wed, 22 Dec 2021'}
plt.savefig(plot_dir+'/corr_wind_spd_column_temperature.png', metadata=figdata)
# Show
plt.show()

#Average of 16 nearby columns
plot_four_avg_col(sst_col_corr, atm_var='wind_spd_atm_T', ocn_var='wind_spd_ocn_T')
plt.suptitle('Correlation between surface wind speed and temperature, averaged over 16 neighboring columns')
plt.tight_layout()
# Save
figdata = {'Title': 'Correlation between surface wind speed and temperature, averaged over 16 neighboring columns', 
           'Author': 'Zofia C Stanley',
           'Description': 'Correlations calculated in north pacific (160E,40N), tropical pacific (130W,0N), south pacific (90W,30S), and indian ocean (60E,20S). Each plot represents the average correlations in 16 neighboring columns.',
           'Creation Time': 'Wed, 22 Dec 2021'}
plt.savefig(plot_dir+'/corr_wind_spd_column_temperature_avg.png', metadata=figdata)
# Show
plt.show()


## Plot correlation of sst with T2m, Wind Speed, and Total Precip.
plt_show = True
plt_save = False
fig, axes = plt.subplots(3,1,figsize=(6, 12), subplot_kw={'projection': ccrs.Mollweide()})
plot_global_field(axes[0], ds['lon'], ds['lat'], sst_t2m_corr, vmin=-0.6, vmax=0.6, add_land_mask=True, ylabel='T2m')
plot_global_field(axes[1], ds['lon'], ds['lat'], sst_windspd_corr, vmin=-0.3, vmax=0.3, add_land_mask=True, ylabel='Wind Speed')
plot_global_field(axes[2], ds['lon'], ds['lat'], sst_tprcp_corr, vmin=-0.4, vmax=0.4, add_land_mask=True, ylabel='Tot. Precip.')
plt.tight_layout()
plt.suptitle('Correlation between SST and atm. variables')

# Show plot
if plt_show:
  plt.show()

# Save plot
if plt_save:
  figdata = {'Title': 'Correlation between SST and atm variables',
             'Author': 'Zofia C Stanley', 
             'Description': 'Correlations between SST and T2m, atm. surface wind speed, and total precipitation. Calculated with 80 member ensemble from Henry Winterbottom, interpolated to lat/lon by Wei Huang.',
             'Creation Time': 'Fri, 17 Dec 2021'}
  plt.savefig(plot_dir+'/corr_sst_atm.png', metadata=figdata)

## Plot variance for each field
plt_show = True
plt_save = True
ds_plt = ds_mean.copy()
title = 'Ensemble Mean'
fig, axes = plt.subplots(2,2,figsize=(12, 8), subplot_kw={'projection':ccrs.Mollweide()})
plot_global_field(axes[0,0], ds['lon'], ds['lat'], ds_plt['sst'], vmin=np.nanpercentile(ds_plt['sst'], 1), vmax=np.nanpercentile(ds_plt['sst'], 99), add_land_mask=True, cmap='viridis', ylabel='SST')
plot_global_field(axes[0,1], ds['lon'], ds['lat'], ds_plt['atm_t2m'], vmin=np.nanpercentile(ds_plt['atm_t2m'], 1), vmax=np.nanpercentile(ds_plt['atm_t2m'], 99), cmap='viridis', ylabel='T2m')
plot_global_field(axes[1,0], ds['lon'], ds['lat'], ds_plt['wind_spd'],vmin=np.nanpercentile(ds_plt['wind_spd'], 1), vmax=np.nanpercentile(ds_plt['wind_spd'], 99),   cmap='viridis', ylabel='Wind Speed')
plot_global_field(axes[1,1], ds['lon'], ds['lat'], ds_plt['atm_tprcp'],vmin=np.nanpercentile(ds_plt['atm_tprcp'], 1), vmax=np.nanpercentile(ds_plt['atm_tprcp'], 99), cmap='viridis', ylabel='Tot. Precip') 
plt.tight_layout()
plt.suptitle(title)
# Save data
if plt_save:
  figdata = {'Title': 'Standard deviation: SST and atm variables',
             'Author': 'Zofia C Stanley',
             'Description': 'Ensemble mean of SST, T2m, wind speed, and total precipitation in 80 member ensemble. Colormaps saturate at 98th percentile.',     
             'Creation Time': 'Fri, 17 Dec 2021'}
  plt.savefig(plot_dir+'/mean_sst_atm.png', metadata=figdata)
# Show plot
if plt_show:
  plt.show()
