import netCDF4 as nc
import xarray as xr
import numpy as np
import re
from scipy import stats
from scipy.linalg import sqrtm
import statsmodels.api as sm
from load_data_fns import *

## Where are we working
proj_dir = '/work/noaa/gsienkf/zstanley/projects/obs_loc'
my_data_dir = proj_dir +'/my_data/20151206.030000'

## Which locations are we looking at?
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

def get_cov_cor(column_name):
  ds = xr.open_dataset(my_data_dir+'/five_columns_'+column_name+'.nc')
  ## Define relevant dimensions
  num_atm_lev = len(ds['atm_lev'])
  num_ocn_lev = len(ds['ocn_lev'])
  num_tot_lev = num_atm_lev + num_ocn_lev
  num_ens_mem = len(ds['ens_mem'])
  num_bootstrap_mult_factor = 5**2
  num_bs_ens_mem = num_ens_mem * num_bootstrap_mult_factor
  ## Convert dataset to numpy array in an ugly loop
  ds_manual = np.empty((num_tot_lev, num_bs_ens_mem))
  start = 0
  end = 80
  for lon in ds['lon'].values: 
    for lat in ds['lat'].values:
      # Select atm columns
      ds_atm = ds['atm_T'].sel(lon=lon, lat=lat, method='nearest')
      ds_atm_np = ds_atm.to_numpy()
      ds_atm_np = np.swapaxes(ds_atm_np, 0, 1)
      ens_atm_avg = ds_atm_np.mean(axis=1)
      ens_atm_avg_expand = np.tile(ens_atm_avg, (num_ens_mem,1)).transpose()
      # Select ocn columns
      ds_ocn = ds['ocn_Temp'].sel(lon=lon, lat=lat, method='nearest')
      ds_ocn_np = ds_ocn.to_numpy()
      ds_ocn_np = np.swapaxes(ds_ocn_np, 0, 1)
      ens_ocn_avg = ds_ocn_np.mean(axis=1)
      ens_ocn_avg_expand = np.tile(ens_ocn_avg, (num_ens_mem,1)).transpose()
      # Store atm and ocn columns
      ds_manual[:num_atm_lev , start:end] = ds_atm_np - ens_atm_avg_expand
      ds_manual[num_atm_lev:num_tot_lev , start:end] = ds_ocn_np - ens_ocn_avg_expand
      start = start + 80
      end = end + 80
  ## Compute covariance matrix
  cov_temp = np.cov(ds_manual)
  cor_temp = np.corrcoef(ds_manual)
  return cov_temp, cor_temp

def save_cov_cor(these_columns, save_dir):
  how_many_cols = len(these_columns['lons'])
  for i in range(how_many_cols-1):
    print(i)
    save_name = these_columns['save_name'][i]
    cov_temp, cor_temp = get_cov_cor(save_name)
    ## Save covariance and correlation matrices
    np.save(save_dir+'/'+save_name+'_cov.npy', cov_temp)
    np.save(save_dir+'/'+save_name+'_cor.npy', cor_temp)


save_cov_cor(which_columns2, my_data_dir)


