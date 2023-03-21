"""Given true and ensemble BH^T and HBH^T, compute error in Kalman Gain."""

import warnings
import numpy as np

from observationoperator import PointObserver



class KalmanGainComputer():
    """Take BH^T and HBH^T matrices from an observation operator (H) and compute:
    
        1. True K = BH^T (HBH^T + R)
        2. Localized ensemble K
        3. Error between localized ensemble K and True K.
    
    Attributes:
        R (float): observation error variance, here set equal to true HBH^T
        true_K (array): true Kalman gain, computed with true HBH^T and BH^T
    """
    
    
    def __init__(self, obs):
        """Sets R matrix and true Kalman gain associated with a specific observation
        
        Args:
            obs (PointObserver): stores true and ensemble BH^T and HBH^T for a single column
        """
        
        self.set_R(obs)
        self.set_true_K(obs)
    
    
    
    def __call__(self, obs, **kwargs):
        """Returns error in Kalman gain with optional localization specified through kwargs
        
        Args:
            obs (PointObserver): stores true and ensemble BH^T and HBH^T for a single column
            kwargs:
                loc_weight_HBHT (float): weight to multiply times HBH^T
                loc_weight_BHT (array): weights to multiply times BH^T
                loc_weight_R (float): weight to multiply times R
                level (int or slice): vertical level where K is to be computed
            
        Returns:
            error (float): RMS distance between ensemble and true Kalman gain
        """
        
        return self.compute_error(obs, **kwargs)
    
    
    
    def set_R(self, obs):
        """Sets observation error variance, R.
        
        Args:
            obs (PointObserver): stores true and ensemble BH^T and HBH^T for a single column
        
        Sets Attribute:
            R (float): observation error variance, here set equal to true HBH^T
        """
        self.R = obs.true_HBHT
    
    
    
    def set_true_K(self, obs):
        """Sets true Kalman gain.
        
        Args:
            obs (PointObserver): stores true and ensemble BH^T and HBH^T for a single column
            
        Sets Attribute:
            true_K (array): true Kalman gain, computed with true HBH^T and BH^T
        """
        self.true_K = self.kalman_gain(obs.true_BHT, obs.true_HBHT, self.R)
        
    
    
    def compute_error_true_K(self, **kwargs):
        
        if 'level' in kwargs:
            level = kwargs.get('level')
            true_K = self.true_K[level]
        else:
            true_K = self.true_K
        
        error = np.mean(np.square(true_K))
        
        return error
    
    
    
    
    def compute_ens_K(self, obs, loc_weight_HBHT=1, loc_weight_BHT=1, loc_weight_R=1, level = None, **kwargs):
        """Computes ensemble Kalman gain with optional localization, for whole column or a single vertical level. 
        
        Args:
            obs (PointObserver): stores true and ensemble BH^T and HBH^T for a single column
            loc_weight_HBHT (float, optional): weight to multiply times HBH^T
            loc_weight_BHT (array, optional): weights to multiply times BH^T
            loc_weight_R (float, optional): weight to multiply times R
            level (int or slice, optional): vertical level where K is to be computed
        
        Returns:
            ens_K (array): array of ensemble Kalman gains
        """
        ens_HBHT  = loc_weight_HBHT * obs.ens_HBHT
        loc_R     = loc_weight_R    * self.R
        
        if level is None:
            ens_BHT   = loc_weight_BHT * obs.ens_BHT
        else:
            ens_BHT   = loc_weight_BHT * obs.ens_BHT[level, :]
        
        ens_K = self.kalman_gain(ens_BHT, ens_HBHT, loc_R)
        
        return ens_K

 
    
    def compute_error(self, obs, **kwargs):
        """Computes RMS distance between (localized) ensemble Kalman gain and true Kalman gain.
        
        Args:
            obs (PointObserver): stores true and ensemble BH^T and HBH^T for a single column
            kwargs:
                loc_weight_HBHT (float): weight to multiply times HBH^T
                loc_weight_BHT (array): weights to multiply times BH^T
                loc_weight_R (float): weight to multiply times R
                level (int or slice): vertical level where K is to be computed
                by_trial (bool): if True return error for each trial
        
        Returns:
            cost (float): RMS distance between (localized) ensemble Kalman gain and true Kalman gain.
        """
        ens_K = self.compute_ens_K(obs, **kwargs)
        
        if 'level' in kwargs:
            level = kwargs.get('level')
            true_K = self.true_K[level]
        else:
            true_K = self.true_K
        
        if ens_K.ndim == 2 :
            num_trials = ens_K.shape[1]
            true_K = np.tile(true_K, [num_trials, 1]).transpose()
        
        if 'by_trial' in kwargs:
            if kwargs.get('by_trial'):
                cost = np.mean(np.square(true_K - ens_K), 0)
        else:
            cost = np.mean(np.square(true_K - ens_K))

        return cost
            
            
            
    @staticmethod
    def kalman_gain(BHT, HBHT, R):
        """Computes Kalman gain via:
            
            K = BH^T (HBH^T + R)
            
        Returns:
            kalman_gain (array): array storing Kalman gain
        """
        kalman_gain = np.divide(BHT, HBHT + R)
        
        return kalman_gain
        
        
        
        