import re
import numpy as np
from scipy import stats

def get_filename(num):
  path = '/work/noaa/gsienkf/zstanley/projects/obs_loc/data/'
  filenum = f"{num+1:02}"
  filename = 'ens1_0000'+filenum+'.nc'
  return path+filename

def get_filenum(filename):
  match_obj = re.search('(?<=ens1_0000).*(?=\.nc)', filename)
  filenum = int(match_obj.group(0))
  return filenum

def preprocess(ds):
  ''' Add an ensemble member coordinate'''
  dsnew = ds.copy()
  # File name contains ensemble member index
  filename = dsnew.filename
  filenum = get_filenum(filename)
  # Add ens_mem dimension
  dsnew['ens_mem'] = (filenum - 1) # Files are indexed from 1
  dsnew = dsnew.expand_dims('ens_mem').set_coords('ens_mem')
  # Get surface wind speed and sst
  dsnew = get_wind_speed(dsnew)
  dsnew = get_sst(dsnew)
  return dsnew  

def get_wind_speed(ds):
  dsnew = ds.copy()
  u2 = np.square(dsnew['atm_u_srf'])
  v2 = np.square(dsnew['atm_v_srf'])
  wind_spd = np.sqrt(u2 + v2)
  dsnew['wind_spd'] = wind_spd
  dsnew.wind_spd.attrs['long_name'] = 'Atmospheric surface wind speed'
  return dsnew

def get_sst(ds):
  dsnew = ds.copy()
  dsnew['sst'] = dsnew.ocn_Temp.sel(ocn_lev=1)
  dsnew.sst.attrs['long_name'] = 'Sea surface temperature'
  return dsnew

