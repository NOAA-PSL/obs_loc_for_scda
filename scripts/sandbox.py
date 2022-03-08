#import netCDF4 as nc
#import xarray as xr
#from cartopy import config
#import cartopy.crs as ccrs
#import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
#import re
from scipy import stats
#from scipy.linalg import sqrtm
#import statsmodels.api as sm
#from load_data_fns import *
from letkf import *

## Set seed
np.random.seed(6513)

## Where are we working
proj_dir = '/work/noaa/gsienkf/zstanley/projects/obs_loc'
data_dir = proj_dir + '/data'
plot_dir = proj_dir + '/plots'

## Load covariance and correlation matrices
cov_temp = np.load(proj_dir+'/my_data/indian_ocean_cov.npy')
cor_temp = np.load(proj_dir+'/my_data/indian_ocean_cor.npy')

## Load matrix square roots
cov_temp_sqrt = np.load(proj_dir+'/my_data/indian_ocean_cov_sqrt.npy')
cov_temp_ocn_sqrt = np.load(proj_dir+'/my_data/indian_ocean_cov_ocn_sqrt.npy')
cov_temp_atm_sqrt = np.load(proj_dir+'/my_data/indian_ocean_cov_atm_sqrt.npy')

## Get sizes
ens_size = 20
num_levs_atm = cov_temp_atm_sqrt.shape[0]
num_levs_ocn = cov_temp_ocn_sqrt.shape[0]
num_levs_tot = cov_temp.shape[0]
this_num_levs = num_levs_ocn
this_cov_sqrt = cov_temp_ocn_sqrt
this_cov = this_cov_sqrt @ this_cov_sqrt.transpose()

## Set observation operator
HofX = np.zeros((1, this_num_levs))
HofX[0, 0] = 1

## Generate ensemble
rnd = np.random.normal(size=(this_num_levs, ens_size))
x_ens = this_cov_sqrt @ rnd

## Generate synthetic observations
xt = this_cov_sqrt @ np.random.normal(size=(this_num_levs, 1))
y_ob = HofX @  xt #this_cov_sqrt @ np.random.normal(xt, scale=1/10, size=xt.shape)
R = HofX @ this_cov @ HofX.transpose()
R_inv = np.reciprocal(R)

## Get state and observation positions and localization radii
ocn_z = np.load(proj_dir+'/my_data/ocn_z.npy')
atm_p = np.load(proj_dir+'/my_data/atm_p.npy')
pos_state = ocn_z
pos_ob = pos_state[0]

## Look a single column update
loc_rad = 50
loc_rad_state = np.full_like(pos_state, loc_rad)
loc_rad_ob = np.full_like(pos_ob, loc_rad)
## Get analysis mean
analysis_mean, analysis_cov = letkf(x_ens, HofX, R_inv, y_ob, pos_state, pos_ob, loc_rad_state, loc_rad_ob, inflate=1)

## Calculate analysis increments
background_mean = np.mean(x_ens, 1)
true_increment = xt[:,0] - background_mean
da_increment = analysis_mean - background_mean
innovation = y_ob - HofX @ background_mean

## Plot increment and observation
plt.plot(true_increment[:37], pos_state[:37], label='True increment')
plt.plot(da_increment[:37], pos_state[:37], label='DA increment')
plt.plot(innovation, pos_ob, marker='o', label='Observation innovation')
plt.legend()
plt.xlabel('Increment (K)')
plt.ylabel('Depth (m)')
plt.title('Increment with localization radius = '+str(loc_rad))
plt.savefig(plot_dir+'/ocn_increment_150m')
plt.show()


## Run several trials
ntrial = 1000
loc_rads = [1, 5, 7, 10, 20, 50, 100, 250, 5e9]
n_loc_rad = len(loc_rads)

xa_err = np.empty([ntrial, n_loc_rad])

for ind_trial in range(ntrial):
  for ind_loc in range(n_loc_rad):
    # Get localization radius
    loc_rad = loc_rads[ind_loc]
    loc_rad_state = np.full_like(pos_state, loc_rad)
    loc_rad_ob = np.full_like(pos_ob, loc_rad)
    ## Get analysis mean
    analysis_mean, analysis_cov = letkf(x_ens, HofX, R_inv, y_ob, pos_state, pos_ob, loc_rad_state, loc_rad_ob, inflate=1)
    ## Calculate error in analysis mean
    analysis_err = np.sqrt(np.mean(np.square(analysis_mean - xt)))
    ## Store error in the mean
    xa_err[ind_trial, ind_loc] = analysis_err

## Average over trials
xa_err_avg = np.mean(xa_err, 0)

## Plot analysis mean vs localization radius
plt.plot(loc_rads, xa_err_avg, label='Error with localization')
plt.hlines(np.sqrt(R), min(loc_rads), max(loc_rads), color='r', label='Observation error')
plt.legend()
plt.xlabel('Localization radius (m)')
plt.ylabel('Analysis RMSE')
plt.title('Error in the analysis mean: Assim. of SST ob. into ocn. temp.')
plt.tight_layout()
plt.savefig(plot_dir+'/indian_ocean_optimal_loc_rad_ocn')
plt.show()



