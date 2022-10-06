"""I observe and produce BH^T and HBH^T."""

import warnings
import numpy as np

from ensemblecovariancecomputer import EnsembleCovarianceComputer



class PointObserver():
    """Observation operator for a point-observation. Takes covariance matrices B and returns BH^T and HBH^T.
        
    Attributes:
        obs_name (str): name of observation
        which_fluid (str, optional if obs_name is in pre-defined list): which fluid is observed, options: 'atm', 'ocn'
        obs_level (int, optional if obs_name is pre-defined list): which vertical level is observed (relative to the fluid, not the whole column).
        obs_ind_cpl (int): observation index relative to entire column, 0 at top of atmosphere and increasing to bottom of ocean.
        dist_atm (float): distance from observation to atm levels
        dist_ocn (float): distance from observation to ocn levels
        true_HBHT (float): HBH^T, the diagonal element in the background error covariance matrix associated with the observed level.
        true_BHT (array): BH^T, the row in the background error covariance matrix associated with the observed level.
        ens_HBHT (float): HBH^T, the diagonal element in the ensemble background error covariance matrix associated with the observed level.
        ens_BHT (array): BH^T, the row(s) in the ensemble background error covariance matrix(ces) associated with the observed level.
    """
        
        
    obs_name                = None
    which_fluid             = None
    obs_level               = None
        
        
    defined_obs = {
            
        'ast' : {
            'which_fluid'   : 'atm',
            'obs_level'     : -1     } ,
            
        'sst' : {
            'which_fluid'   : 'ocn',
            'obs_level'     : 0      }
            
    }
       
        
        
    def __init__(self, obs_name, which_fluid=None, obs_level=None):
        """Sets observation name, which fluid is observed, and which level is observed within that fluid.
        
        Note:
            This class holds a list of pre-defined observations. If obs_name is in this list, the pre-defined values will be used.
            Otherwise, which_fluid and obs_level are required in addition to obs_name (which is always required).
        
        Args:
            obs_name (str): name of observation
            which_fluid (str, optional if obs_name is in pre-defined list): which fluid is observed, options: 'atm', 'ocn'
            obs_level (int, optional if obs_name is pre-defined list): which vertical level is observed (relative to the fluid, not the whole column).
        """
        
        self.obs_name = obs_name
        
        if obs_name in self.defined_obs:
            for key, val in self.defined_obs[obs_name].items():
                setattr(self, key, val)
        elif which_fluid not in ['atm', 'ocn']:
            raise TypeError('which_fluid is not correctly specified and obs_name: '+obs_name+' is not in pre-defined list. '
                            'Specify which_fluid (options: \'atm\', \'ocn\'), or set obs_name to one of the following: '+str(list(self.defined_obs.keys())))
        elif obs_level is None:
            raise TypeError('obs_level is not specified and obs_name: '+obs_name+' is not in pre-defined list. '
                            'Specify obs_level, or set obs_name to one of the following: '+str(list(self.defined_obs.keys())))
        else:
            self.which_fluid     = which_fluid
            self.obs_level       = obs_level
        
        
        
    def __call__(self, enscov):
        """
        Args:
            enscov (EnsembleCovarianceComputer): stores true and ensemble covariances for a single column
        """
        
        self.set_obs_ind_cpl(enscov)
        self.set_dist(enscov)
        
        self.set_true_HBHT(enscov)
        self.set_true_BHT(enscov)
        self.set_ens_HBHT(enscov)
        self.set_ens_BHT(enscov)
        
        
        
    def __str__(self):
        mystr = \
                f"EnsembleCovarianceComputer:\n\n"+\
                f"    {'Name of Observation':<24s}: {self.obs_name}\n"+\
                f"    {'Which Fluid is Observed':<24s}: {self.which_fluid}\n"+\
                f"    {'Which Level is Observed':<24s}: {self.obs_level}\n"+\
                f" --- \n"+\
                f"    Pre-defined observation types:\n"+\
                f"        "+str(list(self.defined_obs.keys()))+"\n"
        
        return mystr

    

    def __repr__(self):
        return self.__str__()
        
    
    
    def set_obs_ind_cpl(self, enscov):
        """Sets observation index relative to the entire column.
        
        Sets Attribute:
            obs_ind_cpl (int): observation index relative to entire column, 0 at top of atmosphere and increasing to bottom of ocean.
        """
        
        if self.which_fluid == 'atm':
            
            if self.obs_level >= 0:
                self.obs_ind_cpl = self.obs_level
            else:
                self.obs_ind_cpl = enscov.len_atm + self.obs_level
                
        elif self.which_fluid == 'ocn':
            
            if self.obs_level >= 0:
                self.obs_ind_cpl = enscov.len_atm + self.obs_level
            else:
                self.obs_ind_cpl = enscov.len_cpl + self.obs_level
                
        else:
            raise TypeError('which_fluid must be set to either \'atm\' or \'ocn\'.')
           
        
        
    def set_dist(self, enscov):
        """Sets distance from observation to different vertical levels in atm and ocn
        
        Args:
            enscov (EnsembleCovarianceComputer): stores true and ensemble covariances for a single column
        
        Sets Attributes:
            dist_atm (float): distance from observation to atm levels
            dist_ocn (float): distance from observation to ocn levels
        """
        
        if self.which_fluid == 'atm':
            self.dist_atm = np.abs(np.log(enscov.atm_p) - np.log(enscov.atm_p[self.obs_level]))
            self.dist_ocn = np.abs(enscov.ocn_z - enscov.ocn_z[0])
            
        elif self.which_fluid == 'ocn':
            self.dist_atm = np.abs(np.log(enscov.atm_p) - np.log(enscov.atm_p[-1]))
            self.dist_ocn = np.abs(enscov.ocn_z - enscov.ocn_z[self.obs_level])
            
        else:
            raise TypeError('which_fluid must be set to either \'atm\' or \'ocn\'.')
            
        
        
    def set_true_HBHT(self, enscov):
        """Given true covariance matrix, sets true HBH^T.
        
        Args:
            enscov (EnsembleCovarianceComputer): stores true and ensemble covariances for a single column
        
        Sets Attributes:
            true_HBHT (float): HBH^T, the diagonal element in the background error covariance matrix associated with the observed level.
        """
        self.true_HBHT = enscov.cov_cpl[self.obs_ind_cpl, self.obs_ind_cpl]
        
    
    
    def set_true_BHT(self, enscov):
        """Given true covariance matrix, sets true BH^T.
        
        Args:
            enscov (EnsembleCovarianceComputer): stores true and ensemble covariances for a single column
        
        Sets Attributes:
            true_BHT (array): BH^T, the row in the background error covariance matrix associated with the observed level.
        """
        self.true_BHT = enscov.cov_cpl[:, self.obs_ind_cpl]
    
    
    
    def set_ens_HBHT(self, enscov):
        """Given ensemble covariance matrix(ces), sets ensemble HBH^T.
        
        Args:
            enscov (EnsembleCovarianceComputer): stores true and ensemble covariances for a single column
        
        Sets Attributes:
            ens_HBHT (float): HBH^T, the diagonal element in the ensemble background error covariance matrix associated with the observed level.
        """
        self.ens_HBHT = enscov.ens_cov_cpl[self.obs_ind_cpl, self.obs_ind_cpl, :]

        
    
    def set_ens_BHT(self, enscov):
        """Given ensemble covariance matrix(ces), sets ensemble BH^T.
        
        Args:
            enscov (EnsembleCovarianceComputer): stores true and ensemble covariances for a single column
        
        Sets Attributes:
            ens_BHT (array): BH^T, the row(s) in the ensemble background error covariance matrix(ces) associated with the observed level.
        """
        self.ens_BHT = enscov.ens_cov_cpl[:, self.obs_ind_cpl, :]
        

        



        

