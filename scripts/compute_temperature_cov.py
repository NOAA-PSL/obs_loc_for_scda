import netCDF4 as nc
import xarray as xr
from cartopy import config
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import re
from scipy import stats
from scipy.linalg import sqrtm
import statsmodels.api as sm
from load_data_fns import *

## Where are we working
proj_dir = '/work/noaa/gsienkf/zstanley/projects/obs_loc'
data_dir = proj_dir + '/data'
plot_dir = proj_dir + '/plots'

## Open netcdf files with xarray
ds = xr.open_mfdataset(data_dir+'/ens1*.nc', autoclose=True, preprocess=preprocess)

## Select 25 columns in the Indian Ocean
lon_sel_pt = 80
lat_sel_pt = -25 
lon_sel = slice(lon_sel_pt-2, lon_sel_pt+3, 1)
lat_sel = slice(lat_sel_pt-2, lat_sel_pt+3, 1)  
ds_sel = ds['atm_T'].sel(lon=lon_sel, lat=lat_sel)

## Convert to numpy array so that we can easily compute covariance matrix
ds_np = ds_sel.to_numpy()
ds_np = np.swapaxes(ds_np, 0, 1)
ds_np = ds_np.reshape((64, 2000))

## Check for normality
level = 64
data_points = (ds_manual[level,]-np.mean(ds_manual[level,]))/np.std(ds_manual[level,])
sm.qqplot(data_points, line='45')
plt.title('Atmospheric surface temperature is approximately normally distributed')
plt.show()
#plt.savefig(plot_dir+'/atm_srf_temp_is_normal.png')

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
for lon in range(lon_sel_pt-2, lon_sel_pt+3, 1):
  for lat in range(lat_sel_pt-2, lat_sel_pt+3, 1):
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

## Save covariance and correlation matrices
np.save(proj_dir+'/my_data/indian_ocean_cov.npy', cov_temp)
np.save(proj_dir+'/my_data/indian_ocean_cor.npy', cor_temp)

## Compute matrix square roots
cov_temp = np.load(proj_dir+'/my_data/indian_ocean_cov.npy')
cov_temp_sqrt = np.linalg.cholesky(cov_temp)
cov_temp_atm_sqrt = np.linalg.cholesky(cov_temp[:num_atm_lev, :num_atm_lev])
cov_temp_ocn_sqrt = np.linalg.cholesky(cov_temp[num_atm_lev:, num_atm_lev:])
np.save(proj_dir+'/my_data/indian_ocean_cov_sqrt.npy', cov_temp_sqrt)
np.save(proj_dir+'/my_data/indian_ocean_cov_ocn_sqrt.npy', cov_temp_ocn_sqrt)
np.save(proj_dir+'/my_data/indian_ocean_cov_atm_sqrt.npy', cov_temp_atm_sqrt)

## Plot covariance matrix
plt.imshow(cov_temp)
plt.colorbar()
plt.title('Atmospheric and Oceanic Temperature Covariance Matrix')
#plt.show()
plt.savefig(plot_dir+'/atm_ocn_temperature_cov.png')

## Plot covariance matrix
plt.imshow(cor_temp, vmin=-1, vmax=1, cmap='bwr')
plt.colorbar()
plt.title('Atmospheric and Oceanic Temperature Correlation Matrix')
#plt.show()
plt.savefig(plot_dir+'/atm_ocn_temperature_cor.png')
plt.show()

## Plot std. dev. of each level
spread = np.sqrt(np.diag(cov_temp))
plt.plot(spread, np.arange(num_tot_lev-1, -1, -1))
plt.axhline(y=74.5)
plt.xlabel('Spread (std. dev.)')
plt.ylabel('Vertical Level')
plt.title('Spread in 80 member ensemble, Indian Ocean')
plt.savefig(plot_dir+'/indian_ocean_80_mem_ens_spread.png')
plt.show()

# Save ocean vertical coordinates
ocn_z = -1 * ds['ocn_lev'].to_numpy()

# Save atm vertical coordinates
ds_full_sel = ds.sel(lon=lon_sel, lat=lat_sel)
atm_delp = ds_full_sel['atm_delp'].mean(['ens_mem','lat', 'lon'])
atm_p = atm_delp.cumsum(dim='atm_lev').to_numpy()
atm_z = ds_full_sel['atm_DZ'].mean(['ens_mem', 'lat', 'lon']).cumsum(dim='atm_lev').to_numpy()

np.save(proj_dir+'/my_data/ocn_z.npy', ocn_z)
np.save(proj_dir+'/my_data/atm_p.npy', atm_p)
