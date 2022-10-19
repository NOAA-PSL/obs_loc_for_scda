import itertools
import time
import warnings
import sys
#from dask import delayed
#from dask.distributed import Client
import xarray as xr
import numpy as np
from scipy import linalg
from scipy import optimize

from ensemblecovariancecomputer import EnsembleCovarianceComputer
from observationoperator import PointObserver
from errorcomputer import ErrorComputer



## Where are we working
proj_dir = '/work/noaa/gsienkf/zstanley/projects/obs_loc'
data_dir = '/work2/noaa/gsienkf/weihuang/WCLEKF_PRODFORECAST/20151205000000/latlongrid-20151206.030000/AtmOcnIce'
my_data_dir = proj_dir +'/my_data/20151206.030000'


  
#@delayed
def compute_loc(ds):
    ''' Get optimal localization radii for a given lat/lon 
    returns ErrorComputer objects ec_ast, ec_sst in that order
    '''
    #lat = lat_lon[0]
    #lon = lat_lon[1]
    
    # Pull out column
    #ds = ds.sel(lat=lat, lon=lon)
    
    # Check for ocean columns by looking at nans or zeros
    ast_sst_corr = ds['cov_atm_ocn'].sel(atm_lev=126, ocn_lev=1).values
    
    if np.isnan(ast_sst_corr) or ast_sst_corr == 0 :
        return [None, None]
    
    else:
        enscov =  EnsembleCovarianceComputer()
        enscov(ds)
        
        ast = PointObserver('ast')
        ast(enscov)
        
        sst = PointObserver('sst')
        sst(enscov)
        
        ec_ast = ErrorComputer(enscov)
        ec_ast(ast)
        
        ec_sst = ErrorComputer(enscov)
        ec_sst(sst)
        
        return [ec_ast, ec_sst]

    
    
def main():
    
    ## Get input argument
    arg_lat = int(sys.argv[1])
    arg_lon = int(sys.argv[2])    

    ## Open averaged covariances
    ds = xr.open_dataset(my_data_dir+'/temperature_covariances_averaged.nc', chunks={'lat':2, 'lon':36})
    
    # Store lat/lon pairs
    lats = ds['lat'].values[arg_lat:arg_lat+3]
    lons = ds['lon'].values[arg_lon:arg_lon+18]
    lat_lon_list = list(itertools.product(lats,lons))
    
    ## Select these lats
    lat_slice = slice(lats[0], lats[-1], 1)
    lon_slice = slice(lons[0], lons[-1], 1)
    ds = ds.sel(lat=lat_slice, lon=lon_slice)
    
    ## Initialize results
    results = [ [None] * 2] * (len(lat_lon_list))

    ## Stage computations
    for ind in range(len(lat_lon_list)):
        lat = lat_lon_list[ind][0]
        lon = lat_lon_list[ind][1]
        # Pull out column
        ds_sel = ds.sel(lat=lat, lon=lon)
        results[ind] = compute_loc(ds_sel)

    ## Initialize empty data arrays for optimal localization radius for (ast, sst) x (atm, ocn)
    keys = ['error_unloc_atm', 'error_unloc_ocn', 'locrad_gcr_atm', 'locrad_gcr_ocn', 'error_gcr_atm', \
            'error_gcr_ocn', 'locrad_gcra_atm', 'locrad_gcra_ocn', 'locatten_gcra_atm',\
            'locatten_gcra_ocn', 'error_gcra_atm', 'error_gcra_ocn', 'error_eorl_atm', 'error_eorl_ocn']
    
    keys_ast = [key+'_ast' for key in keys]
    keys_sst = [key+'_sst' for key in keys]   
    
    ds['error_unloc_atm_ast'] = xr.zeros_like(ds['ocn_z'].min('ocn_lev'))
    ds['error_unloc_atm_sst'] = xr.zeros_like(ds['error_unloc_atm_ast'])
    
    for key in keys[1:]:
        ds[key+'_ast'] = xr.zeros_like(ds['error_unloc_atm_ast'])
        ds[key+'_sst'] = xr.zeros_like(ds['error_unloc_atm_ast'])

    
    # Save localization radii
    for ind in range(len(lat_lon_list)):
        
        lat = lat_lon_list[ind][0]
        lon = lat_lon_list[ind][1]
        
        if results[ind][0] is not None:
            for key in keys:
                ds[key+'_ast'].loc[dict(lat=lat, lon=lon)] = results[ind][0].__dict__[key]
                ds[key+'_sst'].loc[dict(lat=lat, lon=lon)] = results[ind][1].__dict__[key]

        ds[keys_ast+keys_sst].to_netcdf(my_data_dir+'/opt_loc_'+str(arg_lat)+'_'+str(arg_lon)+'.nc')
    
    
    
if __name__ == '__main__':
    tic = time.perf_counter()
    main()
    toc = time.perf_counter()
    print(f"Computed optimal loc in {toc - tic:0.4f} seconds", flush=True)
    
