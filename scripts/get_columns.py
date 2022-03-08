import netCDF4 as nc
import xarray as xr
import numpy as np
import re
from scipy import stats
from load_data_fns import *

## Where are we working
proj_dir = '/work/noaa/gsienkf/zstanley/projects/obs_loc'
data_dir = proj_dir + '/data'
plot_dir = proj_dir + '/plots'
my_data_dir = proj_dir +'/my_data'

## Open netcdf files with xarray
ds = xr.open_mfdataset(data_dir+'/ens1*.nc', autoclose=True, preprocess=preprocess)

## Get 2-d correlations
sst_t2m_corr = xr.corr(ds['sst'], ds['atm_t2m'], dim='ens_mem')

## Get columns
def get_column(ds, lon, lat):
  col_lon = slice(lon-2, lon+2, 1)
  col_lat = slice(lat-2, lat+2, 1)
  column = ds.sel(lon=col_lon, lat=col_lat)
  return column

## Pick columns
which_columns = {
  'lons' : [-174.5, 45.5, 75.5, -129.5, 160.5],
  'lats' : [-29.5, -59.5, -24.5, 0.5, 40.5],
  'name' : ['South Pacific', 'Southern Ocean', 'Indian Ocean', 'Tropical Pacific', 'North Pacific'],
  'save_name' : ['south_pacific', 'southern_ocean', 'indian_ocean', 'tropical_pacific', 'north_pacific']
}

def save_columns(ds, these_columns, save_dir):
  how_many_cols = len(these_columns['lons'])
  for i in range(how_many_cols):
    lon = these_columns['lons'][i]
    lat = these_columns['lats'][i]
    name = these_columns['save_name'][i]
    filename = save_dir+'/five_columns_'+name+'.nc'
    columns = get_column(ds, lon=lon, lat=lat)
    columns.to_netcdf(filename) 

save_columns(ds, which_columns, my_data_dir)
