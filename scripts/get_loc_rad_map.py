import itertools
import time
from dask import delayed
from dask.distributed import Client
import xarray as xr
import numpy as np
from scipy import linalg
from scipy import optimize
from letkf import gaspari_cohn

## Where are we working
proj_dir = '/work/noaa/gsienkf/zstanley/projects/obs_loc'
data_dir = '/work2/noaa/gsienkf/weihuang/WCLEKF_PRODFORECAST/20151205000000/latlongrid-20151206.030000/AtmOcnIce'
my_data_dir = proj_dir +'/my_data/20151206.030000'

        
class CovColumn:
    #
    def __init__(self, ds, ens_size=20, num_trials=1000):
        # Set parameters
        self.ens_size = ens_size
        self.num_trials = num_trials
        # Set dimension size
        self.len_atm = ds.atm_lev.shape[0]
        self.len_ocn = ds.ocn_lev.shape[0]
        self.len_cpl = self.len_atm + self.len_ocn
        # set observation indices
        self.ind_ast = self.len_atm - 1
        self.ind_sst = self.ind_ast + 1
        # set covariance
        self._set_full_cov(ds)
        self.set_dist(ds)
    #   
    def _set_full_cov(self, ds):
        # convert to numpy
        cov_atm_atm = ds['cov_atm_atm'].to_numpy()
        cov_atm_ocn = ds['cov_atm_ocn'].to_numpy()
        cov_ocn_ocn = ds['cov_ocn_ocn'].to_numpy()
        # form full matrix
        full_cov = np.empty((self.len_cpl, self.len_cpl))
        full_cov[:self.len_atm, :self.len_atm] = cov_atm_atm
        full_cov[:self.len_atm, self.len_atm:] = cov_atm_ocn
        full_cov[self.len_atm:, :self.len_atm] = cov_atm_ocn.transpose()
        full_cov[self.len_atm:, self.len_atm:] = cov_ocn_ocn
        # set cov
        self.cov_atm = cov_atm_atm
        self.cov_ocn = cov_ocn_ocn
        self.cov_cpl = full_cov
    #   
    def set_dist(self, ds):
        atm_p = ds['atm_p']
        ocn_z = ds['ocn_z']
        self.dist_ocn = ocn_z - ocn_z[0]
        self.dist_atm = np.abs(np.log(atm_p) - np.log(atm_p[-1]))
    #
    def _make_cov_pos_def(self):
        min_eval = np.min(linalg.eigh(self.cov_cpl, eigvals_only=True))
        if min_eval < -1e6:
            raise ValueError('Matrix is /really/ not positive definite with min eval = '+ str(min_eval))
        elif min_eval < 0 :
            print('Matrix is not positive definite. Smallest eigenvalue is ' + str(min_eval))
            self.cov_cpl = self.cov_cpl + (-1*min_eval + 1e-13) * np.eye(self.len_cpl)
        elif ( min_eval > 0 & min_eval < 1e-13):
            
    #        
    def set_cov_sqrt(self):
        self._make_cov_pos_def()
        self.cov_sqrt = np.linalg.cholesky(self.cov_cpl)
    #   
    def _compute_ens_cov(self):
        rnd = np.random.normal(size=(self.len_cpl, self.ens_size))
        x_err_ens = self.cov_sqrt @ rnd
        ens_cov = np.cov(x_err_ens)
        return ens_cov
    #
    def _compute_multiple_ens_cov(self):
        ens_cov_mats = np.empty([self.len_cpl, self.len_cpl, self.num_trials])
        for ii in range(self.num_trials):
            this_ens_cov_mat = self._compute_ens_cov()
            ens_cov_mats[:, :, ii] = this_ens_cov_mat
        return ens_cov_mats
    #
    def set_ens_covs(self):
        # set ensemble covariances
        cov_cols = self._compute_multiple_ens_cov()
        self.ens_cov_ast_atm = cov_cols[:self.len_atm, self.ind_ast, :]
        self.ens_cov_ast_ocn = cov_cols[self.len_atm:, self.ind_ast, :]
        self.ens_cov_sst_atm = cov_cols[:self.len_atm, self.ind_sst, :]
        self.ens_cov_sst_ocn = cov_cols[self.len_atm:, self.ind_sst, :]
    #
    def costfn_gc(self, loc_rad, true_cov, ens_covs, dist):
        ''' Error in covariance matrix'''
        loc = gaspari_cohn(dist/(loc_rad/2))
        loc_expand = np.tile(loc, [self.num_trials, 1]).transpose()
        true_corr_expand = np.tile(true_cov, [self.num_trials, 1]).transpose()
        cost = np.mean(np.square(true_corr_expand - loc_expand * ens_covs))
        return cost
    #
    def set_optimal_loc_rads(self):
        self.set_cov_sqrt()
        self.set_ens_covs()
        # minimize cost functions
        self.locrad_gc_ast_atm = optimize.minimize_scalar(self.costfn_gc, args=(self.cov_cpl[:self.len_atm, self.ind_ast], self.ens_cov_ast_atm, self.dist_atm)).x
        self.locrad_gc_ast_ocn = optimize.minimize_scalar(self.costfn_gc, args=(self.cov_cpl[self.len_atm:, self.ind_ast], self.ens_cov_ast_ocn, self.dist_ocn)).x
        self.locrad_gc_sst_atm = optimize.minimize_scalar(self.costfn_gc, args=(self.cov_cpl[:self.len_atm, self.ind_sst], self.ens_cov_sst_atm, self.dist_atm)).x
        self.locrad_gc_sst_ocn = optimize.minimize_scalar(self.costfn_gc, args=(self.cov_cpl[self.len_atm:, self.ind_sst], self.ens_cov_sst_ocn, self.dist_ocn)).x
    #
    def get_optimal_loc_rads(self):
        return [self.locrad_gc_ast_atm, self.locrad_gc_ast_ocn, self.locrad_gc_sst_atm, self.locrad_gc_sst_ocn]
        
@delayed
def get_loc_rads_for_lat_lon(lat_lon, ds):
    ''' Get optimal localization radii for a given lat/lon 
    returns loc rad for [ ast_atm, ast_ocn, sst_atm, sst_ocn ] in that order
    '''
    lat = lat_lon[0]
    lon = lat_lon[1]
    # Pull out column
    column = ds.sel(lat=lat, lon=lon)
    # Check for ocean columns by looking at nans
    if np.isnan(column['cov_atm_ocn'].sel(atm_lev=126, ocn_lev=1).values):
        return [np.nan, np.nan, np.nan, np.nan]
    else:
        cov_col = CovColumn(column)
        cov_col.set_optimal_loc_rads()
        result = [cov_col.locrad_gc_ast_atm, cov_col.locrad_gc_ast_ocn, cov_col.locrad_gc_sst_atm, cov_col.locrad_gc_sst_ocn]
        return result

def main():
    ## Open averaged covariances
    ds = xr.open_dataset(my_data_dir+'/temperature_covariances_averaged.nc')
    #
    # Store lat/lon pairs
    lats = ds['lat'].values
    lons = ds['lon'].values
    lat_lon_list = list(itertools.product(lats,lons))
    #
    ## Initialize results
    results = [ [None] * 4] * (len(lat_lon_list))
    #
    ## Stage computations
    for ind in range(len(lat_lon_list)):
            results[ind] = get_loc_rads_for_lat_lon(lat_lon_list[ind], ds)
    #
    ## Execute results with Dask
    with Client() as c:
        results = c.compute(results, sync=True)
    #
    ## Initialize empty data arrays for optimal localization radius for (ast, sst) x (atm, ocn)
    ds['loc_rad_gc_ast_atm'] = xr.zeros_like(ds['ocn_z'].min('ocn_lev'))
    ds['loc_rad_gc_ast_ocn'] = xr.zeros_like(ds['loc_rad_gc_ast_atm'])
    ds['loc_rad_gc_sst_atm'] = xr.zeros_like(ds['loc_rad_gc_ast_atm'])
    ds['loc_rad_gc_sst_ocn'] = xr.zeros_like(ds['loc_rad_gc_ast_atm'])
    #
    # Save localization radii
    for ind in range(len(lat_lon_list)):
        lat = lat_lon_list[ind][0]
        lon = lat_lon_list[ind][1]
        ds['loc_rad_gc_ast_atm'].loc[dict(lat=lat, lon=lon)] = results[ind][0]
        ds['loc_rad_gc_ast_ocn'].loc[dict(lat=lat, lon=lon)] = results[ind][1]
        ds['loc_rad_gc_sst_atm'].loc[dict(lat=lat, lon=lon)] = results[ind][2]
        ds['loc_rad_gc_sst_ocn'].loc[dict(lat=lat, lon=lon)] = results[ind][3]
    #
    ds[['loc_rad_gc_ast_atm','loc_rad_gc_ast_ocn','loc_rad_gc_sst_atm','loc_rad_gc_sst_ocn']].to_netcdf(my_data_dir+'/loc_rad_gc.nc')
                
if __name__ == '__main__':
    tic = time.perf_counter()
    main()
    toc = time.perf_counter()
    print(f"Computed optimal loc in {toc - tic:0.4f} seconds")
    
