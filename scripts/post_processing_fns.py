import numpy as np
from letkf import *

def dens_wright_eos(T, S, p):
  """
  Equation of state for sea water given by Wright, 1997, J. Atmos. Ocean. Tech., 14, 735-740.
  Units: T[degC],S[PSU],p[Pa]
  Returns density [kg m-3]
  """
  a0, a1, a2 = 7.057924e-4, 3.480336e-7, -1.112733e-7
  b0, b1, b2, b3, b4, b5 = 5.790749e8, 3.516535e6, -4.002714e4, 2.084372e2, 5.944068e5, -9.643486e3
  c0, c1, c2, c3, c4, c5 = 1.704853e5, 7.904722e2, -7.984422, 5.140652e-2, -2.302158e2, -3.079464

  al0 = a0 + a1*T + a2*S
  p0  = b0 + b4*S + T * (b1 + T*(b2 + b3*T) + b5*S)
  l = c0 + c4*S + T * (c1 + T*(c2 + c3*T) + c5*S)
  rho = (p + p0) / (l + al0*(p+p0))
  
  return rho

def compute_analysis_err_one_domain(observe_this_level, localization_radii, this_cov, this_cov_sqrt, pos_state, ens_size = 20, start=0, stop=37):
    ## Set observation operator
    num_levs = this_cov.shape[0]
    HofX = np.zeros((1, num_levs))
    HofX[0, observe_this_level] = 1
    
    ## Set observation position
    pos_ob = pos_state[observe_this_level]
    
    ## Generate synthetic observations
    xt = this_cov_sqrt @ np.random.normal(size=(num_levs, 1))
    y_ob = HofX @ this_cov_sqrt @ np.random.normal(xt, scale=1, size=xt.shape) # obs error set to equal background error
    
    ## Generate ensemble
    rnd = np.random.normal(size=(num_levs, ens_size))
    x_ens = this_cov_sqrt @ rnd
    P_ens = np.cov(x_ens)
    
    ## Set observation error variance equal to true variance
    R = HofX @ this_cov @ HofX.transpose()
    R_inv = np.reciprocal(R)
    
    ## Calculate analysis increments
    background_mean = np.mean(x_ens, 1)
    innovation = y_ob - HofX @ background_mean
    kf_increment = ( this_cov @ HofX.transpose() /(HofX @ this_cov @ HofX.transpose() + R)) * innovation 
    
    ## Get analysis mean
    num_loc_rads =  len(localization_radii)
    analysis_error = np.empty(num_loc_rads)
    da_increment = np.empty((num_levs, num_loc_rads))
    for loc_ind in range(num_loc_rads):
        # Set localization radius
        localization_radius = localization_radii[loc_ind]
        loc_rad_state = np.full_like(pos_state, localization_radius)
        loc_rad_ob = np.full_like(pos_ob, localization_radius)
        # Get analysis mean
        analysis_mean = letkf_one_domain(x_ens, HofX, R_inv, y_ob, pos_state, pos_ob, loc_rad_state, loc_rad_ob, inflate=1)[0]
        this_increment = analysis_mean - background_mean
        da_increment[:, loc_ind] = this_increment
        # Get analysis error
        analysis_error[loc_ind] = np.sqrt( np.mean( np.square( this_increment[start:stop] - kf_increment[start:stop,0]) ) )
    
    return analysis_error, da_increment, kf_increment, innovation

def run_multiple_trials_one_domain(observe_this_level, localization_radii, this_cov, this_cov_sqrt, pos_state, num_trials = 100, start=0, stop=37):
    errs = np.empty((len(localization_radii), num_trials))
    for trial in range(num_trials):
        analysis_error = compute_analysis_err_one_domain(observe_this_level=observe_this_level, localization_radii=localization_radii, this_cov=this_cov, this_cov_sqrt=this_cov_sqrt, pos_state=pos_state, start=start, stop=stop)[0]
        errs[:, trial] = analysis_error
    avg_err = np.mean(errs, axis=1)
    return avg_err