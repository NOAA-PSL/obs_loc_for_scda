import netCDF4 as nc
import xarray as xr
import numpy as np
from scipy import stats
from load_data_fns import *
from letkf import *

## Where are we working
proj_dir = '/work/noaa/gsienkf/zstanley/projects/obs_loc'
#data_dir = '/work2/noaa/gsienkf/weihuang/WCLEKF_PRODFORECAST/20151205000000/production/latlongrid-20151206.030000/AtmOcnIce'
data_dir = '/work2/noaa/gsienkf/weihuang/WCLEKF_PRODFORECAST/20151205000000/latlongrid-20151206.030000/AtmOcnIce'
my_data_dir = proj_dir +'/my_data/20151206.030000'

## Open netcdf files with xarray
ds = xr.open_mfdataset(data_dir+'/ens1*.nc', autoclose=True, preprocess=preprocess)
ds['atm_p'] = ds['atm_delp'].mean('ens_mem').cumsum(dim='atm_lev') / 1e2
lats = ds['lat'].values[::5] #get every fifth lat value
lons = ds['lon'].values[::5]

## Get columns
def get_column(ds, lon, lat):
  col_lon = slice(lon-2, lon+2, 1)
  col_lat = slice(lat-2, lat+2, 1)
  column = ds.sel(lon=col_lon, lat=col_lat)
  return column

## Check land mask
def check_land_mask(column):
    mask = np.sum(column['atm_slmsk'].values)
    return (mask==0)

def get_atm_pos(column):
    atm_p = column['atm_p'].mean(['lat', 'lon']).to_numpy()
    pos_state_atm = np.abs(np.log(atm_p) - np.log(atm_p[-1]))
    pos_ob_atm = pos_state_atm[-1]
    return pos_state_atm, pos_ob_atm

def get_column_mean(ds):
    # initialize array
    num_atm_levs = len(ds['atm_lev'])
    num_ocn_levs = len(ds['ocn_lev'])
    num_tot_levs = num_atm_levs + num_ocn_levs
    ds_manual = np.empty(num_tot_levs)
    # Get mean of atm bootstrapped ensemble
    ds_atm = ds['atm_T'].mean(['lat', 'lon', 'ens_mem'])
    ds_atm = ds_atm.to_numpy()
    # Get mean of ocn bootstrapped ensemble
    ds_ocn = ds['ocn_Temp'].mean(['lat', 'lon', 'ens_mem'])
    ds_ocn = ds_ocn.to_numpy()
    # Store atm and ocn temperature mean
    ds_manual[:num_atm_levs] = ds_atm - 273.15
    ds_manual[num_atm_levs:num_tot_levs] = ds_ocn
    ds_manual = np.atleast_2d(ds_manual).transpose()
    return ds_manual

def get_column_cov(ds):
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
  return cov_temp

def create_distance_matrix(spatial_locations):
    N = len(spatial_locations) 
    # Create distance matrix
    dis = np.zeros((N,N))
    for jj in range(N):
        for ii in range(N):
            d = np.abs(spatial_locations[ii]-spatial_locations[jj])
            dis[ii, jj] = d
    return dis

def compute_relative_error(ind_state, ind_ob, loc_rads_state, loc_rad_ob, this_avg, this_cov, this_cov_sqrt, pos_state, pos_ob, ens_size = 20, use_loc=True, ind_top_of_fluid=0):
    ## Set observation operator
    num_levs = this_cov.shape[0]
    HofX = np.zeros((1, num_levs))
    HofX[0, ind_ob] = 1
    ## Generate synthetic observations
    xt = this_avg + this_cov_sqrt @ np.random.normal(scale=1, size=(num_levs, 1)) 
    y_ob = HofX @ ( xt + this_cov_sqrt @ np.random.normal(scale=1, size=(num_levs, 1))) # obs error set to equal 'perfect' background error
    ## Generate ensemble
    rnd = np.random.normal(size=(num_levs, ens_size))
    x_err_ens = this_cov_sqrt @ rnd
    P_ens = np.cov(x_err_ens)
    x_ens = this_avg + x_err_ens
    ## Form background observation perturbations
    y_ens = HofX @ x_ens
    ## Set observation error variance equal to true variance
    R = HofX @ this_cov @ HofX.transpose()
    R_inv = np.reciprocal(R)
    ## Calculate background mean and innovation
    background_mean = np.mean(x_ens, 1)
    innovation = y_ob - HofX @ background_mean
    ## Compute perfect analysis
    perfect_kg = ( this_cov @ HofX.transpose() /(HofX @ this_cov @ HofX.transpose() + R))
    ## Take only state
    x_ens = x_ens[ind_state]
    background_mean = background_mean[ind_state]
    perfect_kg = perfect_kg[ind_state, 0]
    pos_state = pos_state[(ind_state - ind_top_of_fluid)]
    ## Get relative error for each localization radius
    num_loc_rads =  len(loc_rads_state)
    norm_rel_error_letkf = np.empty(num_loc_rads)
    for loc_ind in range(num_loc_rads):
        # Get localization radius
        localization_radius = loc_rads_state[loc_ind]
        # LETKF update
        loc_rad_state = np.full_like(pos_state, localization_radius)
        analysis_mean_letkf = letkf(x_ens, y_ens, R_inv, y_ob, pos_state, pos_ob, loc_rad_state, loc_rad_ob, inflate=1, use_loc=use_loc)[0]
        letkf_kg = (analysis_mean_letkf - background_mean)/innovation
        # Compute relative error
        kg_err_letkf = perfect_kg - letkf_kg
        # Store relative error
        norm_rel_error_letkf[loc_ind] = np.sqrt(np.sum(np.square(kg_err_letkf))) / np.sqrt(np.sum(np.square(perfect_kg)))
    return norm_rel_error_letkf

def run_multiple_trials_rel_err(ind_state, ind_ob, loc_rads_state, loc_rad_ob, this_avg, this_cov, this_cov_sqrt, pos_state, pos_ob, ens_size = 20, num_trials=100, use_loc=True, ind_top_of_fluid=0):
    norm_rel_errors_letkf = np.empty((len(loc_rads_state), num_trials))
    for trial in range(num_trials):
        norm_rerr_letkf = compute_relative_error(ind_state=ind_state, ind_ob=ind_ob, loc_rads_state=loc_rads_state, loc_rad_ob=loc_rad_ob, this_avg=this_avg, this_cov=this_cov, this_cov_sqrt=this_cov_sqrt, pos_state=pos_state, pos_ob=pos_ob, ens_size=ens_size, use_loc=use_loc, ind_top_of_fluid=ind_top_of_fluid)
        norm_rel_errors_letkf[:, trial] = norm_rerr_letkf
    return norm_rel_errors_letkf

## Define a range of localization radii for ocn and atm
loc_rads_ocn = [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220]
loc_rads_atm = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2]
loc_rad_noloc = 1e5

## Get state location for ocean
ocn_z = ds['ocn_lev'].to_numpy()
pos_state_ocn = ocn_z - ocn_z[0]
pos_ob_ocn = pos_state_ocn[0]

## Get indices of relevant pressure levels in atm and depth levels in ocn
# atmosphere
num_atm_levs = len(ds['atm_lev'])
ind_ast = num_atm_levs - 1
ind100hPa = np.argmin(np.abs(ds['atm_p'].mean(['lat', 'lon']).values-100))
# ocean
ind0m = num_atm_levs
ind_sst = ind0m
ind2km_ocn  = np.argmin(np.abs(ds['ocn_lev'].values-2000))
num_ocn_levs = len(ds['ocn_lev'])
# both
ind2km = ind0m + ind2km_ocn
num_tot_levs = num_atm_levs + num_ocn_levs

## Define range of each fluid over which to compute error
ind_atm = np.arange(ind100hPa, num_atm_levs)
ind_ocn = np.arange(ind0m, ind2km)

## Define top of fluid
ind_top_of_fluid_atm = 0
ind_top_of_fluid_ocn = ind0m

## Number of trials
num_trials=20

## Initialize empty arrays
save_locs_ast_atm = np.zeros((lats.size, lons.size)) #localization radius (LETKF)
save_locs_sst_atm = np.zeros((lats.size, lons.size)) #localization radius (LETKF)
save_locs_ast_ocn = np.zeros((lats.size, lons.size)) #localization radius (LETKF)
save_locs_sst_ocn = np.zeros((lats.size, lons.size)) #localization radius (LETKF)
save_errs_ast_atm = np.zeros((lats.size, lons.size)) #relative error
save_errs_sst_atm = np.zeros((lats.size, lons.size)) #relative error
save_errs_ast_ocn = np.zeros((lats.size, lons.size)) #relative error
save_errs_sst_ocn = np.zeros((lats.size, lons.size)) #relative error

## Save localization radius for each (lat,lon) pair
for lat_ind in range(lats.size):
    print(lat_ind)
    for lon_ind in range(lons.size):
        lat = lats[lat_ind]
        lon = lons[lon_ind]
        # get column
        column = get_column(ds, lon%360, lat)
        # check land mask
        if check_land_mask(column):
            # get average temperature
            this_avg = get_column_mean(column)
            # get covariance
            this_cov = get_column_cov(column) # this is way too slow
            this_cov_sqrt = np.linalg.cholesky(this_cov)
            # get atm state positions
            pos_state_atm, pos_ob_atm = get_atm_pos(column)
            # Get errors
            # AST into ATM
            err_ast_atm = run_multiple_trials_rel_err(ind_atm, ind_ast, loc_rads_atm, loc_rad_noloc, this_avg, this_cov, this_cov_sqrt, pos_state_atm, pos_ob_atm, num_trials=num_trials, use_loc=True, ind_top_of_fluid=ind_top_of_fluid_atm)
            err_ast_atm = np.median(err_ast_atm, axis=1)
            min_ind = np.argmin(err_ast_atm)
            save_locs_ast_atm[lat_ind, lon_ind] = loc_rads_atm[min_ind]
            save_errs_ast_atm[lat_ind, lon_ind] = err_ast_atm[min_ind]
            # SST into ATM
            err_sst_atm = run_multiple_trials_rel_err(ind_atm, ind_sst, loc_rads_atm, loc_rad_noloc, this_avg, this_cov, this_cov_sqrt, pos_state_atm, pos_ob_atm, num_trials=num_trials, use_loc=True, ind_top_of_fluid=ind_top_of_fluid_atm)
            err_sst_atm = np.median(err_sst_atm, axis=1)
            min_ind = np.argmin(err_sst_atm)
            save_locs_sst_atm[lat_ind, lon_ind] = loc_rads_atm[min_ind]
            save_errs_sst_atm[lat_ind, lon_ind] = err_sst_atm[min_ind]
            # AST into OCN
            err_ast_ocn = run_multiple_trials_rel_err(ind_ocn, ind_ast, loc_rads_ocn, loc_rad_noloc, this_avg, this_cov, this_cov_sqrt, pos_state_ocn, pos_ob_ocn, num_trials=num_trials, use_loc=True, ind_top_of_fluid=ind_top_of_fluid_ocn)
            err_ast_ocn = np.median(err_ast_ocn, axis=1)
            min_ind = np.argmin(err_ast_ocn)
            save_locs_ast_ocn[lat_ind, lon_ind] = loc_rads_ocn[min_ind]
            save_errs_ast_ocn[lat_ind, lon_ind] = err_ast_ocn[min_ind]
            # SST into OCN
            err_sst_ocn = run_multiple_trials_rel_err(ind_ocn, ind_sst, loc_rads_ocn, loc_rad_noloc, this_avg, this_cov, this_cov_sqrt, pos_state_ocn, pos_ob_ocn, num_trials=num_trials, use_loc=True, ind_top_of_fluid=ind_top_of_fluid_ocn)
            err_sst_ocn = np.median(err_sst_ocn, axis=1)
            min_ind = np.argmin(err_sst_ocn)
            save_locs_sst_ocn[lat_ind, lon_ind] = loc_rads_ocn[min_ind]
            save_errs_sst_ocn[lat_ind, lon_ind] = err_sst_ocn[min_ind]

np.save(my_data_dir+'/save_locs_ast_atm.npy', save_locs_ast_atm)
np.save(my_data_dir+'/save_locs_sst_atm.npy', save_locs_sst_atm)
np.save(my_data_dir+'/save_locs_ast_ocn.npy', save_locs_ast_ocn)
np.save(my_data_dir+'/save_locs_sst_ocn.npy', save_locs_sst_ocn)

np.save(my_data_dir+'/save_errs_ast_atm.npy', save_errs_ast_atm)
np.save(my_data_dir+'/save_errs_sst_atm.npy', save_errs_sst_atm)
np.save(my_data_dir+'/save_errs_ast_ocn.npy', save_errs_ast_ocn)
np.save(my_data_dir+'/save_errs_sst_ocn.npy', save_errs_sst_ocn)