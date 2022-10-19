import xarray as xr
import numpy as np

## Where are we working
proj_dir = '/work/noaa/gsienkf/zstanley/projects/obs_loc'
data_dir = '/work2/noaa/gsienkf/weihuang/WCLEKF_PRODFORECAST/20151205000000/latlongrid-20151206.030000/AtmOcnIce'
my_data_dir = proj_dir +'/my_data/20151206.030000'


def main():
    
    ## Open optimal localization files
    ds = xr.open_mfdataset(my_data_dir+'/opt_loc*.nc')
    
    ## Save files as single dataset
    ds.to_netcdf(my_data_dir+'/opt_loc_global.nc')


if __name__ == '__main__':
    main()