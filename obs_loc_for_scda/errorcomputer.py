"""Find optimal localization schemes and compute errors."""

import numpy as np
from scipy import optimize

from ensemblecovariancecomputer import EnsembleCovarianceComputer
from observationoperator import PointObserver
from kalmangaincomputer import KalmanGainComputer
from localizationfunctions import gaspari_cohn_univariate as gaspari_cohn


# want to pass obs name and state fluid name and then get back errors and localization weights

# 'Practical' localization:
#     1. Cutoff with ensemble correlation??
#     2. Single localization radius
#     3. Each fluid gets one localization radius
#     4. Same as #3, but include attenuation

class ErrorComputer():
    """For a given observation operator, compute:
    
        Error with no localization:
            1. No localization
        Error with 'Optimal' Localization:
            1. EORL: Empirical Optimal R-matrix Localization
            2. GC-R: Optimal Gaspari-Cohn localization length scale
            3. GC-R-A: Same as GC-R, but with an added attenuation factor
    
    Attributes:
        R (float): observation error variance, here set equal to true HBH^T
        true_K (array): true Kalman gain, computed with true HBH^T and BH^T
    """
    
    
    def __init__(self, enscov):
        """Sets various attributes which define dimensions of atm/ocn
        
        Args:
            enscov (EnsembleCovarianceComputer): stores true and ensemble covariances for a single column
            
        Sets Attributes:
            len_atm (int): Number of atmospheric levels
            len_ocn (int): Number of ocean levels
            slice_atm (slice): Slice object containing indices corresponding to atm variables
            slice_ocn (slice): Slice object containing indices corresponding to ocn variables
            num_trials (int): Number of small ensembles that are generated
        """
        
        self.len_atm = enscov.len_atm
        self.len_ocn = enscov.len_ocn
        self.slice_atm = slice(0, enscov.len_atm, 1)
        self.slice_ocn = slice(enscov.len_atm, enscov.len_cpl, 1)
        self.num_trials = enscov.num_trials

        
        
    def __call__(self, obs):
        """Computes unlocalized and optimally localized Kalman Gains and their associated errors
        
        Args:
            obs (PointObserver): stores true and ensemble BH^T and HBH^T for a single column
        """
        
        kg = KalmanGainComputer(obs)
        
        self.compute_unlocalized_error(kg, obs)
        self.compute_optimal_gcr(kg, obs)
        self.compute_optimal_gcra(kg, obs)
        self.compute_optimal_eorl(kg, obs)
    
    
    
    def compute_unlocalized_error(self, kg, obs):
        """Computes error in Kalman Gain with no localization
        
        Args:
            kg (KalmanGainComputer): computes error in Kalman gain
            obs (PointObserver): stores true and ensemble BH^T and HBH^T for a single column
            
        Sets Attributes:
            error_unloc_atm (float): error in unlocalized Kalman gain in atm
            error_unloc_ocn (float): error in unlocalized Kalman gain in ocn
        """
        self.error_unloc_atm = kg(obs, level=self.slice_atm)
        self.error_unloc_ocn = kg(obs, level=self.slice_ocn)
            

    
    def compute_optimal_gcr(self, kg, obs):
        """Computes optimal Gaspri-Cohn localization radius (no attenuation factor)
        
        Args:
            kg (KalmanGainComputer): computes error in Kalman gain
            obs (PointObserver): stores true and ensemble BH^T and HBH^T for a single column
            
        Sets Attributes:
            error_gcr_atm (float): error in optimal GC-R Kalman gain in atm
            error_gcr_ocn (float): error in optimal GC-R Kalman gain in ocn
        """
        
        
        result_atm = optimize.minimize_scalar(self.cost_gcr, args=(kg, obs, obs.dist_atm, self.slice_atm, self.num_trials), method='brent', options={'xtol':1e-4})
        result_ocn = optimize.minimize_scalar(self.cost_gcr, args=(kg, obs, obs.dist_ocn, self.slice_ocn, self.num_trials), method='brent', options={'xtol':1e-4})
        
        self.locrad_gcr_atm = result_atm.x
        self.locrad_gcr_ocn = result_ocn.x
        
        self.error_gcr_atm = self.cost_gcr(self.locrad_gcr_atm, kg, obs, obs.dist_atm, self.slice_atm, self.num_trials)
        self.error_gcr_ocn = self.cost_gcr(self.locrad_gcr_ocn, kg, obs, obs.dist_ocn, self.slice_ocn, self.num_trials)
        
    

    def compute_optimal_gcra(self, kg, obs):
        
        result_atm = optimize.minimize(self.cost_gcra, x0=[self.locrad_gcr_atm, 1], args=(kg, obs, obs.dist_atm, self.slice_atm, self.num_trials), method='nelder-mead', options={'xatol':1e-2, 'fatol':1e-4})
        result_ocn = optimize.minimize(self.cost_gcra, x0=[self.locrad_gcr_ocn, 1], args=(kg, obs, obs.dist_ocn, self.slice_ocn, self.num_trials), method='nelder-mead', options={'xatol':1e-2, 'fatol':1e-4})
        
        self.locrad_gcra_atm = result_atm.x[0]
        self.locrad_gcra_ocn = result_ocn.x[0]
        
        self.locatten_gcra_atm = result_atm.x[1]
        self.locatten_gcra_ocn = result_ocn.x[1]
        
        self.error_gcra_atm = self.cost_gcra(result_atm.x, kg, obs, obs.dist_atm, self.slice_atm, self.num_trials)
        self.error_gcra_ocn = self.cost_gcra(result_ocn.x, kg, obs, obs.dist_ocn, self.slice_ocn, self.num_trials)
        
    
    
    def compute_optimal_eorl(self, kg, obs):
        
        locweight_eorl_atm = np.zeros(self.len_atm)
        locweight_eorl_ocn = np.zeros(self.len_ocn)
        
        for level in range(self.len_atm):
            locweight_eorl_atm[level] = optimize.minimize_scalar(self.cost_eorl, args=(kg, obs, level), options={'xtol':1e-4}).x
                                                                 
        for level in range(self.len_ocn):
            locweight_eorl_ocn[level] = optimize.minimize_scalar(self.cost_eorl, args=(kg, obs, self.len_atm + level), options={'xtol':1e-4}).x
            
        #self.locweight_eorl_atm = locweight_eorl_atm
        #self.locweight_eorl_ocn = locweight_eorl_ocn
        
        locweight_eorl_atm = np.tile(locweight_eorl_atm, [self.num_trials, 1]).transpose()
        locweight_eorl_ocn = np.tile(locweight_eorl_ocn, [self.num_trials, 1]).transpose()

        self.error_eorl_atm = kg(obs, loc_weight_R = locweight_eorl_atm, level = self.slice_atm)
        self.error_eorl_ocn = kg(obs, loc_weight_R = locweight_eorl_ocn, level = self.slice_ocn)

        
        
    @staticmethod
    def cost_gcr(loc_rad, kg, obs, dist, level, num_trials):
        """Computes error in Kalman Gain with Gaspri-Cohn localization (no attenuation factor)
        
        Args:
            loc_rad (float): localization radius
            kg (KalmanGainComputer): computes error in Kalman gain
            obs (PointObserver): stores true and ensemble BH^T and HBH^T for a single column
            dist (array): distance from observation to vertical levels
            level (int or slice): which vertical levels are considered
            
        Returns:
            cost (float): error in Kalman gain
        """
        loc = np.divide(1, gaspari_cohn(dist, (loc_rad/2)))
        loc = np.tile(loc, [num_trials, 1]).transpose()
        cost = kg(obs, loc_weight_R = loc, level = level)
        return cost
    
    
    
    @staticmethod
    def cost_gcra(loc_params, kg, obs, dist, level, num_trials):
        """Computes error in Kalman Gain with Gaspri-Cohn localization and an attenuation factor
        
        Args:
            loc_params (list): [localization radius, attenuation factor]
            kg (KalmanGainComputer): computes error in Kalman gain
            obs (PointObserver): stores true and ensemble BH^T and HBH^T for a single column
            dist (array): distance from observation to vertical levels
            level (int or slice): which vertical levels are considered
            
        Returns:
            cost (float): error in Kalman gain
        """
        loc_rad = loc_params[0]
        loc_atten = loc_params[1]
        loc = np.divide(1, loc_atten * gaspari_cohn(dist, (loc_rad/2)))
        loc = np.tile(loc, [num_trials, 1]).transpose()
        cost = kg(obs, loc_weight_R = loc, level = level)
        return cost
    
    
    
    @staticmethod
    def cost_eorl(loc_weight_R, kg, obs, level):
        """Computes error in Kalman Gain with given localization weight
        
        Args:
            loc_weight_R (float): localization weight for R-matrix localization
            kg (KalmanGainComputer): computes error in Kalman gain
            obs (PointObserver): stores true and ensemble BH^T and HBH^T for a single column
            level (int or slice): which vertical levels are considered
            
        Returns:
            cost (float): error in Kalman gain
        """
        cost = kg(obs, loc_weight_R = loc_weight_R, level = level)
        return cost
    
    
    
