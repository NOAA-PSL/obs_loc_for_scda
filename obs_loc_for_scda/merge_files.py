import xarray as xr
import numpy as np

## Where are we working
proj_dir = '/work/noaa/gsienkf/zstanley/projects/obs_loc'
data_dir = '/work2/noaa/gsienkf/weihuang/WCLEKF_PRODFORECAST/20151205000000/latlongrid-20151206.030000/AtmOcnIce'
my_data_dir = proj_dir +'/my_data/20151206.030000'


def merge_by_lat():
    
    for lat in range(0, 180, 3):
        ## Open optimal localization files
        ds = xr.open_mfdataset(my_data_dir+'/opt_loc_'+str(lat)+'_*.nc', autoclose=True, parallel=True)
        
        ## Save files as single dataset
        ds.load().to_netcdf(my_data_dir+'/opt_loc_lat_'+str(lat)+'.nc')


def merge_all():

    ds = xr.open_mfdataset(my_data_dir+'/opt_loc_lat_*.nc', autoclose=True, parallel=True)

    ds.load().to_netcdf(my_data_dir+'/opt_loc_global.nc')


if __name__ == '__main__':
    merge_by_lat()
    merge_all()

