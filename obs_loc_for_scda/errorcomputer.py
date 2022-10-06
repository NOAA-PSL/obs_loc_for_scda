import warnings
import numpy as np
from scipy import optimize

from ensemblecovariancecomputer import EnsembleCovarianceComputer
from observationoperator import PointObserver
from kalmangaincomputer import KalmanGainComputer
from localizationfunctions import gaspari_cohn_univariate as gaspari_cohn


# want to pass obs name and state fluid name and then get back errors and localization weights
# No localization:
#     1. No localization -- done
# 'Optimal' Localization:
#     1. EORL -- done
#     2. GC-R -- done
#     3. GC-R-A -- done
#     4. BUMP
# 'Practical' localization:
#     1. Cutoff with ensemble correlation??
#     2. Single localization radius
#     3. Each fluid gets one localization radius
#     4. Same as #3, but include attenuation

class ErrorComputer():
    
    
    def __init__(self, enscov):
        self.len_atm = enscov.len_atm
        self.len_ocn = enscov.len_ocn
        self.slice_atm = slice(0, enscov.len_atm, 1)
        self.slice_ocn = slice(enscov.len_atm, enscov.len_cpl, 1)
        self.num_trials = enscov.num_trials

        
        
    def __call__(self, obs):
        
        kg = KalmanGainComputer(obs)
        
        self.compute_unlocalized_error(kg, obs)
        self.compute_optimal_gcr(kg, obs)
        self.compute_optimal_gcra(kg, obs)
        self.compute_optimal_eorl(kg, obs)
    
    
    
    def cost_gcr(self, loc_rad, kg, obs, dist, level):
        loc = np.divide(1, gaspari_cohn(dist, (loc_rad/2)))
        loc = np.tile(loc, [self.num_trials, 1]).transpose()
        cost = kg(obs, loc_weight_R = loc, level = level)
        return cost
    
    
    
    def cost_gcra(self, loc_params, kg, obs, dist, level):
        loc_rad = loc_params[0]
        loc_atten = loc_params[1]
        loc = np.divide(1, loc_atten * gaspari_cohn(dist, (loc_rad/2)))
        loc = np.tile(loc, [self.num_trials, 1]).transpose()
        cost = kg(obs, loc_weight_R = loc, level = level)
        return cost
    
    
    
    def cost_eorl(self, loc_weight_R, kg, obs, level):
        cost = kg(obs, loc_weight_R = loc_weight_R, level = level)
        return cost
    
    
    
    def compute_unlocalized_error(self, kg, obs):
        self.error_unloc_atm = kg(obs, level=self.slice_atm)
        self.error_unloc_ocn = kg(obs, level=self.slice_ocn)
            

    
    def compute_optimal_gcr(self, kg, obs):
        
        result_atm = optimize.minimize_scalar(self.cost_gcr, args=(kg, obs, obs.dist_atm, self.slice_atm))
        result_ocn = optimize.minimize_scalar(self.cost_gcr, args=(kg, obs, obs.dist_ocn, self.slice_ocn))
        
        self.locrad_gcr_atm = result_atm.x
        self.locrad_gcr_ocn = result_ocn.x
        
        self.error_gcr_atm = self.cost_gcr(self.locrad_gcr_atm, kg, obs, obs.dist_atm, self.slice_atm)
        self.error_gcr_ocn = self.cost_gcr(self.locrad_gcr_ocn, kg, obs, obs.dist_ocn, self.slice_ocn)
        
    

    def compute_optimal_gcra(self, kg, obs):
        
        result_atm = optimize.minimize(self.cost_gcra, x0=[self.locrad_gcr_atm, 1], args=(kg, obs, obs.dist_atm, self.slice_atm), method='nelder-mead')
        result_ocn = optimize.minimize(self.cost_gcra, x0=[self.locrad_gcr_ocn, 1], args=(kg, obs, obs.dist_ocn, self.slice_ocn), method='nelder-mead')
        
        self.locrad_gcra_atm = result_atm.x[0]
        self.locrad_gcra_ocn = result_ocn.x[0]
        
        self.locatten_gcra_atm = result_atm.x[1]
        self.locatten_gcra_ocn = result_ocn.x[1]
        
        self.error_gcra_atm = self.cost_gcra(result_atm.x, kg, obs, obs.dist_atm, self.slice_atm)
        self.error_gcra_ocn = self.cost_gcra(result_ocn.x, kg, obs, obs.dist_ocn, self.slice_ocn)
        
    
    
    def compute_optimal_eorl(self, kg, obs):
        
        locweight_eorl_atm = np.zeros(self.len_atm)
        locweight_eorl_ocn = np.zeros(self.len_ocn)
        
        for level in range(self.len_atm):
            locweight_eorl_atm[level] = optimize.minimize_scalar(self.cost_eorl, args=(kg, obs, level)).x
                                                                 
        for level in range(self.len_ocn):
            locweight_eorl_ocn[level] = optimize.minimize_scalar(self.cost_eorl, args=(kg, obs, self.len_atm + level)).x
            
        #self.locweight_eorl_atm = locweight_eorl_atm
        #self.locweight_eorl_ocn = locweight_eorl_ocn
        
        locweight_eorl_atm = np.tile(locweight_eorl_atm, [self.num_trials, 1]).transpose()
        locweight_eorl_ocn = np.tile(locweight_eorl_ocn, [self.num_trials, 1]).transpose()

        self.error_eorl_atm = kg(obs, loc_weight_R = locweight_eorl_atm, level = self.slice_atm)
        self.error_eorl_ocn = kg(obs, loc_weight_R = locweight_eorl_ocn, level = self.slice_ocn)


    
    
