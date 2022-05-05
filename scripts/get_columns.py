import netCDF4 as nc
import xarray as xr
import numpy as np
import re
from scipy import stats
from load_data_fns import *

## Where are we working
proj_dir = '/work/noaa/gsienkf/zstanley/projects/obs_loc'
#data_dir = proj_dir + '/data'
data_dir = '/work2/noaa/gsienkf/weihuang/WCLEKF_PRODFORECAST/20151205000000/production/latlongrid-20151206.030000/AtmOcnIce'
#my_data_dir = proj_dir +'/my_data'
my_data_dir = proj_dir +'/my_data/20151206.030000'

## Open netcdf files with xarray
ds = xr.open_mfdataset(data_dir+'/ens1*.nc', autoclose=True, preprocess=preprocess)

## Get 2-d correlations
sst_t2m_corr = xr.corr(ds['sst'], ds['atm_t2m'], dim='ens_mem')
sst_t2m_corr.to_netcdf(my_data_dir+'/sst_ast_corr.nc')

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

## Pick columns
which_columns2 = {
  'lons' : [-154.5, 35.5, 75.5, -150.5, 160.5],
  'lats' : [-27.5, -49.5, -31.5, 12.5, 40.5],
  'name' : ['South Pacific', 'Southern Ocean', 'Indian Ocean', 'Tropical Pacific', 'North Pacific'],
  'save_name' : ['south_pacific2', 'southern_ocean2', 'indian_ocean2', 'tropical_pacific2', 'north_pacific2']
}

def save_columns(ds, these_columns, save_dir):
  how_many_cols = len(these_columns['lons'])
  for i in range(2,how_many_cols-2):
    print(i)
    lon = these_columns['lons'][i] % 360 # lon is in [0,360] in the data set
    lat = these_columns['lats'][i]
    name = these_columns['save_name'][i]
    filename = save_dir+'/five_columns_'+name+'.nc'
    columns = get_column(ds, lon=lon, lat=lat)
    columns.to_netcdf(filename) 

save_columns(ds, which_columns2, my_data_dir)
