import xarray as xr
from cartopy import config
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt


def plot_global_field(ax, lon, lat, field, vmin, vmax, add_land_mask=False, cmap='bwr', ylabel='', xlabel=''):
  im = ax.pcolormesh(lon, lat, field, vmin=vmin, vmax=vmax, cmap=cmap, transform=ccrs.PlateCarree())
  ax.coastlines()
  if add_land_mask:
    ax.add_feature(cfeature.NaturalEarthFeature('physical', 'land', '50m', edgecolor='black', facecolor='silver'))
  ax.gridlines(draw_labels=False)
  ax.set_ylabel(ylabel)
  ax.set_xlabel(xlabel)
  ax.set_yticks([])
  ax.set_xticks([])
  plt.colorbar(im, ax=ax)
  return(ax)

def plot_single_vert_col(ax, ds, lon, lat, atm_var='sst_atm_T', ocn_var='sst_ocn_T', title=''):
  plt_atm_x = ds[atm_var].sel(lon=lon, lat=lat, method='nearest')
  plt_atm_y = ds['atm_z'].sel(lon=lon, lat=lat, method='nearest')
  plt_ocn_x = ds[ocn_var].sel(lon=lon, lat=lat, method='nearest')
  plt_ocn_y = ds['ocn_z']
  ax.plot(plt_atm_x, 1+plt_atm_y, 'ko')
  ax.plot(plt_ocn_x, 1+plt_ocn_y, 'ko')
  ax.vlines(x=0, ymin=plt_ocn_y.min(), ymax = plt_atm_y.max(), linestyles='dashed')
  ax.hlines(y=0, xmin=-0.5, xmax=1.0, linestyles='dashed')
  plt.xlim([-0.5, 1.1])
  plt.yscale('symlog')
  ax.set_xlabel('correlation')
  ax.set_title(title)
  return(ax)

def plot_avg_vert_col(ax, ds, lon, lat, atm_var='sst_atm_T', ocn_var='sst_ocn_T', title=''):
  plt_lon = slice(lon-2, lon+2, 1)
  plt_lat = slice(lat-2, lat+2, 1)
  plt_atm_x = ds[atm_var].sel(lon=plt_lon, lat=plt_lat).mean(dim=('lat', 'lon'))
  plt_atm_y = ds['atm_z'].sel(lon=plt_lon, lat=plt_lat).mean(dim=('lat', 'lon'))
  plt_ocn_x = ds[ocn_var].sel(lon=plt_lon, lat=plt_lat).mean(dim=('lat', 'lon'))
  plt_ocn_y = ds['ocn_z']
  ax.plot(plt_atm_x, 1+plt_atm_y, 'ko')
  ax.plot(plt_ocn_x, 1+plt_ocn_y, 'ko')
  ax.vlines(x=0, ymin=plt_ocn_y.min(), ymax = plt_atm_y.max(), linestyles='dashed')
  ax.hlines(y=0, xmin=-0.5, xmax=1.0, linestyles='dashed')
  plt.xlim([-0.5, 1.1])
  plt.yscale('symlog')
  ax.set_xlabel('correlation')
  ax.set_title(title)
  return(ax)

def plot_four_single_col(ds, atm_var='sst_atm_T', ocn_var='sst_ocn_T'):
  fig, axes = plt.subplots(1, 4, sharex=True, sharey=True, figsize=(12, 8))
  plot_single_vert_col(axes[0], sst_col_corr, 160, 40, atm_var=atm_var, ocn_var=ocn_var, title='North Pacific')
  plot_single_vert_col(axes[1], sst_col_corr, -130, 0, atm_var=atm_var, ocn_var=ocn_var, title='Tropical Pacific')
  plot_single_vert_col(axes[2], sst_col_corr, -90, -30, atm_var=atm_var, ocn_var=ocn_var, title='South Pacific')
  plot_single_vert_col(axes[3], sst_col_corr, 60, -20, atm_var=atm_var, ocn_var=ocn_var, title='Indian Ocean')
  axes[0].set_ylabel('height')
  return(fig, axes)

def plot_four_avg_col(ds, atm_var='sst_atm_T', ocn_var='sst_ocn_T'):
  fig, axes = plt.subplots(1, 4, sharex=True, sharey=True, figsize=(12, 8))
  plot_avg_vert_col(axes[0], sst_col_corr, 160, 40, atm_var=atm_var, ocn_var=ocn_var, title='North Pacific')
  plot_avg_vert_col(axes[1], sst_col_corr, 360-130, 0, atm_var=atm_var, ocn_var=ocn_var, title='Tropical Pacific')
  plot_avg_vert_col(axes[2], sst_col_corr, 360-90, -30, atm_var=atm_var, ocn_var=ocn_var, title='South Pacific')
  plot_avg_vert_col(axes[3], sst_col_corr, 60, -20, atm_var=atm_var, ocn_var=ocn_var, title='Indian Ocean')
  axes[0].set_ylabel('height')
  return(fig, axes)






