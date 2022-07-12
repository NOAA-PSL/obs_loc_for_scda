import netCDF4 as nc
import xarray as xr
import numpy as np
from multiprocessing import Pool
from os.path import exists
from scipy import stats
from xarray.core.alignment import align
from load_data_fns import *

## Where are we working
proj_dir = '/work/noaa/gsienkf/zstanley/projects/obs_loc'
data_dir = '/work2/noaa/gsienkf/weihuang/WCLEKF_PRODFORECAST/20151205000000/latlongrid-20151206.030000/AtmOcnIce'
my_data_dir = proj_dir +'/my_data/20151206.030000'

## Pick which variables to keep
keep_variables = ['atm_slmsk', 'atm_p', 'ocn_z', 'atm_T', 'ocn_Temp']
keep_ens_variables = ['atm_T', 'ocn_Temp']

def open_full_dataset():
    ## Open netcdf files with xarray
    ds = xr.open_mfdataset(data_dir+'/ens1*.nc', autoclose=True, preprocess=preprocess, parallel=True)
    return ds

def get_vertical_coordinates(ds):
    ## Get vertical coordinates
    ds['atm_p'] = ds['atm_delp'].mean('ens_mem').cumsum(dim='atm_lev') / 1e2
    ds['ocn_z'] = ds['ocn_h'].mean('ens_mem').cumsum(dim='ocn_lev')
    return ds

def clean_ds(ds):
    ## Keep only the desired variables
    ds = ds[keep_variables]
    ## Reduce atm_slmsk along ens_mem dimension
    ds['atm_slmsk'] = ds['atm_slmsk'].sel(ens_mem=0)
    return ds

def save_reduced_dataset(ds):
    ## Save reduced dataset
    ds.to_netcdf(my_data_dir+'/coupled_temperature_fields.nc')
    
def open_reduced_dataset():
    ds = xr.open_dataset(my_data_dir+'/coupled_temperature_fields.nc')
    return ds
    
def reduce_vertical_levels(ds):
    ## Keep only the levels beteween 100 hPa in atm and 500 m in ocean
    these_atm_levs = slice(53, 126, 1) # 100 hPa to surface
    these_ocn_levs = slice(0, 500, 1)  # surface to 500 m
    ds = ds.sel(atm_lev=these_atm_levs, ocn_lev=these_ocn_levs)
    return ds

def save_dataset_with_fewer_vertical_levels(ds):
    ## Save reduced dataset with fewer vertical levels
    ds.to_netcdf(my_data_dir+'/coupled_temperature_fields_100hPa_to_500m.nc')

def open_dataset_with_fewer_vertical_levels():
    ds = xr.open_dataset(my_data_dir+'/coupled_temperature_fields_100hPa_to_500m.nc')
    return ds

def define_ocean_mask(ds):
    ## set mask to 1 where there is ocean and 0 where there is land
    ds['ocn_msk'] = xr.where(np.abs(ds['atm_slmsk'])<1e-10, 1, 0)
    return ds

def compute_covariances(ds, ddof=1, slice_lat=slice(-89.5, 89.5)):
    ''' Compute covarainces between atm temp and ocn temp
    ds:        data set
    ddof:      delta degrees of freedom. ddof=1 (default) gives unbiased estimate
    slice_lat: latitude band over which to do computations 
    '''
    ## 1. Compute mean and perturbations
    ds_mean = ds[keep_ens_variables].mean(dim=['ens_mem'])
    ds_pert = ds[keep_ens_variables] - ds_mean
    # 2. Compute normalized ensemble size
    normalized_count = ds['ens_mem'].size - ddof
    ## 3. Pull out temperature fields
    da_atm = ds_pert['atm_T'].where(ds['ocn_msk'])
    da_ocn = ds_pert['ocn_Temp'].where(ds['ocn_msk'])
    ## 4. Broadcast the two arrays
    da_atm, da_ocn = align(da_atm, da_ocn, join="inner", copy=False)
    ## 5. Slice lat to manually parallelize
    da_atm = da_atm.sel(lat=slice_lat)
    da_ocn = da_ocn.sel(lat=slice_lat)
    ## 6. Compute covariance
    ds_cov = xr.Dataset()
    ds_cov['cov_atm_ocn'] = (da_atm * da_ocn).sum(dim='ens_mem', skipna=False) / normalized_count
    ds_cov['cov_atm_atm'] = (da_atm * da_atm.rename(atm_lev='atm_lev_copy')).sum(dim='ens_mem', skipna=False) / normalized_count
    ds_cov['cov_ocn_ocn'] = (da_ocn * da_ocn.rename(ocn_lev='ocn_lev_copy')).sum(dim='ens_mem', skipna=False) / normalized_count
    ## 7. Add vertical levels to covariance data set
    ds_cov['atm_p'] = ds['atm_p']
    ds_cov['ocn_z'] = ds['ocn_z']
    return ds_cov

def compute_covariances_for_latitude_bands(slice_lat):
    ds_cov = compute_covariances(ds, slice_lat=slice_lat)
    return ds_cov

def compute_covariances_with_pool():
    ## Cut size by three to fit in memory
    latitude_bands = [slice(30.5, 89.5, 1), slice(-29.5, 29.5, 1), slice(-89.5, -30.5, 1)]
    ## Use multiprocessing
    p = Pool(3)
    nh, tr, sh = p.map(compute_covariances_for_latitude_bands, latitude_bands)
    # Merge three lat bands back together
    ds_cov = xr.merge([nh, tr, sh])
    return ds_cov

def save_raw_covariances(ds_cov):
    # Save covariances
    ds_cov.to_netcdf(my_data_dir+'/temperature_covariances_raw.nc')
    
def open_raw_covariances():
    ds_cov = xr.open_dataset(my_data_dir+'/temperature_covariances_raw.nc')
    return ds_cov

def compute_averaged_covariances(ds_cov):
    ## Manually pad the array along the longitudinal direction
    # pad right
    right_slice = slice(358.5, 359.5, 1)
    ds_right = ds_cov.sel(lon=right_slice)
    ds_right['lon'] = ds_right['lon'] - 360
    # pad left
    left_slice = slice(0.5, 1.5, 1)
    ds_left = ds_cov.sel(lon=left_slice)
    ds_left['lon'] = ds_left['lon'] + 360
    # merge pads
    ds_expand = xr.merge([ds_right, ds_cov, ds_left])
    ## Average over neighboring columns
    rolling = ds_expand.rolling({"lat":5,"lon":5}, center=True)
    ds_roll = rolling.construct(lat='window_lat',lon='window_lon')
    ds_cov_avg = ds_roll.mean(dim=['window_lat', 'window_lon'], skipna=False)
    ## Remove padding
    shrink_slice = slice(0.5, 359.5, 1)
    ds_cov_avg = ds_cov_avg.sel(lon=shrink_slice)
    return ds_cov_avg

def save_averaged_covariances(ds_cov_avg):
    ## Save averaged covariances
    rolling_cov.to_netcdf(my_data_dir+'/temperature_covariances_averaged.nc')
    
def main():
    ''' This computes T-T covariances, saving intermediate steps along the way. '''
    
    # If averaged covariances exist, print and exit
    if exists(my_data_dir+'/temperature_covariances_averaged.nc'):
        print('Hooray! Averaged covariances have already been created.')
        
    # If raw covariances exist, average and exit
    elif exists(my_data_dir+'/temperature_covariances_raw.nc'):
        print('Found raw covariances. Proceeding to average and save.')
        # Open raw covariances
        ds_cov = open_raw_covariances()
        # Average covariances
        ds_cov_avg = compute_averaged_covariances(ds_cov)
        save_averaged_covariances(ds_cov_avg)
        
    # If dataset with fewer vertical levels exists, compute raw cov then averaged cov
    elif exists(my_data_dir+'/coupled_temperature_fields_100hPa_to_500m.nc'):
        print('Found data set with reduced vertical levels. Proceeding to compute raw and averaged covariances.')
        # Open dataset
        ds = open_dataset_with_fewer_vertical_levels()
        ds = define_ocean_mask(ds)
        # Compute raw covariances
        ds_cov = compute_covariances_with_pool()
        save_raw_covariances(ds_cov)
        # Average covariances
        ds_cov_avg = compute_averaged_covariances(ds_cov)
        save_averaged_covariances(ds_cov_avg)
        
    # If reduced data set exists, cut vertical levels and compute covariances
    elif exists(my_data_dir+'/coupled_temperature_fields.nc'):
        print('Found data set with reduced variables. Proceeding to compute raw and averaged covariances.')
        # Open reduced dataset and reduce vertical levels
        ds = open_reduced_dataset()
        ds = reduce_vertical_levels(ds)
        save_dataset_with_fewer_vertical_levels(ds)
        # Define ocean mask
        ds = define_ocean_mask(ds)
        # Compute raw covariances
        ds_cov = compute_covariances_with_pool()
        save_raw_covariances(ds_cov)
        # Average covariances
        ds_cov_avg = compute_averaged_covariances(ds_cov)
        save_averaged_covariances(ds_cov_avg)
    
    # Else open the full data set and save all intermediate steps
    else:
        print('Opening full data set. Will save intermediate steps, ending with averaged covariances.')
        # Reduce number of variables
        ds = open_full_dataset()
        ds = get_vertical_coordinates(ds)
        ds = clean_ds(ds)
        save_reduced_dataset(ds)
        # Reduce vertical levels
        ds = reduce_vertical_levels(ds)
        save_dataset_with_fewer_vertical_levels(ds)
        # Define ocean mask
        ds = define_ocean_mask(ds)
        # Compute raw covariances
        ds_cov = compute_covariances_with_pool()
        save_raw_covariances(ds_cov)
        # Average covariances
        ds_cov_avg = compute_averaged_covariances(ds_cov)
        save_averaged_covariances(ds_cov_avg)
        
        
if __name__ == '__main__':
    main()