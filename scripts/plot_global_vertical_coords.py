import netCDF4 as nc
import xarray as xr
import numpy as np
from scipy import stats
from cartopy import config
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
from load_data_fns import *

## Where are we working
proj_dir = '/work/noaa/gsienkf/zstanley/projects/obs_loc'
data_dir = '/work2/noaa/gsienkf/weihuang/WCLEKF_PRODFORECAST/20151205000000/latlongrid-20151206.030000/AtmOcnIce'
my_data_dir = proj_dir +'/my_data/20151206.030000'
plot_dir = proj_dir + '/plots/vertical_coords'

## Open netcdf files with xarray
ds = xr.open_mfdataset(data_dir+'/ens1*.nc', autoclose=True, preprocess=preprocess)
ds['atm_p'] = ds['atm_delp'].mean('ens_mem').cumsum(dim='atm_lev') / 1e2
ds['ocn_z'] = ds['ocn_h'   ].mean('ens_mem').cumsum(dim='ocn_lev')

## Save a map of the world showing atm surface pressure
ax = plt.axes(projection=ccrs.Mollweide())
ax.set_global()
im = ax.pcolormesh(ds['lon'], ds['lat'], ds['atm_p'][-1], vmin=970, vmax = 1030, transform=ccrs.PlateCarree())
ax.add_feature(cfeature.NaturalEarthFeature('physical', 'land', '50m', edgecolor='black', facecolor='silver'))
plt.colorbar(im)
plt.title('Atmospheric Surface Pressure (hPa)')
ax.coastlines(resolution='110m')
ax.gridlines()

# Save
plt.rcParams['figure.figsize'] = [12,6]
plt.savefig(plot_dir+'/atm_surf_p.png')
#plt.show()
plt.close()

## Save a map of the world showing atm surface level thickness
ax = plt.axes(projection=ccrs.Mollweide())
ax.set_global()
im = ax.pcolormesh(ds['lon'], ds['lat'], ds['atm_p'][-1]-ds['atm_p'][-2], vmin=1.2, vmax = 2.6, transform=ccrs.PlateCarree())
ax.add_feature(cfeature.NaturalEarthFeature('physical', 'land', '50m', edgecolor='black', facecolor='silver'))
plt.colorbar(im)
plt.title('Atmospheric Surface Level Thickness (hPa)')
ax.coastlines(resolution='110m')
ax.gridlines()

# Save
plt.rcParams['figure.figsize'] = [12,6]
plt.savefig(plot_dir+'/atm_surf_delp.png')
plt.show()
plt.close()

## Save a map of the world showing atm thickness of bottom 10 levels
ax = plt.axes(projection=ccrs.Mollweide())
ax.set_global()
im = ax.pcolormesh(ds['lon'], ds['lat'], ds['atm_p'][-1]-ds['atm_p'][-11], vmin=30, vmax = 34, transform=ccrs.PlateCarree())
ax.add_feature(cfeature.NaturalEarthFeature('physical', 'land', '50m', edgecolor='black', facecolor='silver'))
plt.colorbar(im)
plt.title('Atmospheric Thickness of Bottom 10 Levels (hPa)')
ax.coastlines(resolution='110m')
ax.gridlines()

# Save
plt.rcParams['figure.figsize'] = [12,6]
plt.savefig(plot_dir+'/atm_bottom10_delp.png')
plt.show()
plt.close()

## Save a map of the world showing ocn surface depth
ax = plt.axes(projection=ccrs.Mollweide())
ax.set_global()
im = ax.pcolormesh(ds['lon'], ds['lat'], ds['ocn_z'][0], vmin=1.9, vmax=2.1, transform=ccrs.PlateCarree())
ax.add_feature(cfeature.NaturalEarthFeature('physical', 'land', '50m', edgecolor='black', facecolor='silver'))
plt.colorbar(im)
plt.title('Ocean Top Level Depth (m)')
ax.coastlines(resolution='110m')
ax.gridlines()

# Save
plt.rcParams['figure.figsize'] = [12,6]
plt.savefig(plot_dir+'/ocn_surf_z.png')
plt.show()
plt.close()

## Save a map of the world showing ocn surface thickness
ax = plt.axes(projection=ccrs.Mollweide())
ax.set_global()
im = ax.pcolormesh(ds['lon'], ds['lat'], ds['ocn_z'][1] - ds['ocn_z'][0], vmin=1.9, vmax=2.1, transform=ccrs.PlateCarree())
ax.add_feature(cfeature.NaturalEarthFeature('physical', 'land', '50m', edgecolor='black', facecolor='silver'))
plt.colorbar(im)
plt.title('Ocean Top Level Thickness (m)')
ax.coastlines(resolution='110m')
ax.gridlines()

# Save
plt.rcParams['figure.figsize'] = [12,6]
plt.savefig(plot_dir+'/ocn_surf_dz.png')
plt.show()
plt.close()

## Save a map of the world showing ocn thickness of top 10 levels
ax = plt.axes(projection=ccrs.Mollweide())
ax.set_global()
im = ax.pcolormesh(ds['lon'], ds['lat'], ds['ocn_z'][10] - ds['ocn_z'][0], vmin=19, vmax=21, transform=ccrs.PlateCarree())
ax.add_feature(cfeature.NaturalEarthFeature('physical', 'land', '50m', edgecolor='black', facecolor='silver'))
plt.colorbar(im)
plt.title('Ocean Thickness of Top 10 Levels (m)')
ax.coastlines(resolution='110m')
ax.gridlines()

# Save
plt.rcParams['figure.figsize'] = [12,6]
plt.savefig(plot_dir+'/ocn_top10_dz.png')
plt.show()
plt.close()


