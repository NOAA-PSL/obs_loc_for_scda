"""Give me true covariances, I give you ensemble covariances"""

import warnings
import xarray as xr
import numpy as np
from scipy import linalg



class EnsembleCovarianceComputer():
    """Compute true and ensemble covariances for a single coupled atm/ocn vertical column.
        
    Attributes:
        ens_size (int): number of ensemble members
        num_trials (int): number of ensembles to be generated
        cov_atm_atm_name (str): name in dataset of covariance which stores atm-atm covariances
        cov_atm_ocn_name (str): name in dataset of covariance which stores atm-ocn covariances
        cov_ocn_ocn_name (str): name in dataset of covariance which stores ocn-ocn covariances
        len_atm (int): number of vertical levels in the atm
        len_ocn (int): number of vertical levels in the ocn
        len_cpl (int): number of vertical levels in the atm + ocn
        cov_cpl (array): covariance matrix
        cov_cpl_sqrt (array): square root of covariance matrix
        ens_cov_cpl (array): n ensemble covariance matrices, where n = num_trials
        atm_p (array): atm pressure levels
        ocn_z (array): ocn z (depth) levels
    """
    
    
    ens_size            = 20
    num_trials          = 1000
    
    cov_atm_atm_name    = 'cov_atm_atm'
    cov_atm_ocn_name    = 'cov_atm_ocn'
    cov_ocn_ocn_name    = 'cov_ocn_ocn'
    
    
    
    @property
    def len_cpl(self):
        try:
            return self._len_cpl
        except AttributeError:
            self._len_cpl = self.len_atm + self.len_ocn
            return self._len_cpl
        
    
    
    @property 
    def cov_cpl_sqrt(self):
        try:
            return self._cov_cpl_sqrt
        except AttributeError:
            self._cov_cpl_sqrt = np.linalg.cholesky(self.cov_cpl)
            return self._cov_cpl_sqrt
    
    
    def __init__(self, **kwargs):
        """All attributes can be changed by passing as keyword arguments to initialization"""
        for key, val in kwargs.items():
            setattr(self, key, val)
    
    
    
    def __call__(self, xds):
        """
        Args:
            xds (:obj:`xarray.Dataset`): dataset storing covariance matrices
        """
        
        self.set_dim_size(xds)

        self.set_full_cov(xds)
        self.make_cov_pos_def()
        
        self.set_ens_cov_cpl()
        
        self.set_vert_coord(xds)
        
        
        
    def __str__(self):
        ecstatus = "set" if hasattr(self,'ens_cov_cpl') else "unset"
        mystr = \
                f"EnsembleCovarianceComputer:\n\n"+\
                f"    Ensemble Covariance Matrices:\n"+\
                f"        status = {ecstatus}\n\n"+\
                f" --- \n"+\
                f"    {'Ensemble Size':<24s}: {self.ens_size}\n"+\
                f"    {'Number of Trials':<24s}: {self.num_trials}\n"

        return mystr

    

    def __repr__(self):
        return self.__str__()

    
        
    def set_dim_size(self, xds):
        """Sets size of atm and ocn domains
        
        Args:
            xds (:obj:`xarray.Dataset`): dataset storing covariance matrices
        
        Sets Attributes:
            len_atm (int): number of vertical levels in the atm
            len_ocn (int): number of vertical levels in the ocn
        """
        
        self.len_atm = xds.atm_lev.shape[0]
        self.len_ocn = xds.ocn_lev.shape[0]
    
    
    
    def set_full_cov(self, xds):
        """Forms full covariance matrix from sub-blocks
        
        Args:
            xds (:obj:`xarray.Dataset`): dataset storing covariance matrices
        
        Sets Attributes:
            cov_cpl (array): covariance matrix
        """

        cov_atm_atm = xds[self.cov_atm_atm_name].to_numpy()
        cov_atm_ocn = xds[self.cov_atm_ocn_name].to_numpy()
        cov_ocn_ocn = xds[self.cov_ocn_ocn_name].to_numpy()
        
        full_cov = np.empty((self.len_cpl, self.len_cpl))
        full_cov[:self.len_atm, :self.len_atm] = cov_atm_atm
        full_cov[:self.len_atm, self.len_atm:] = cov_atm_ocn
        full_cov[self.len_atm:, :self.len_atm] = cov_atm_ocn.transpose()
        full_cov[self.len_atm:, self.len_atm:] = cov_ocn_ocn

        self.cov_cpl = full_cov
        
        
        
    def make_cov_pos_def(self):
        """Adjusts covariance to be positive definite. Raises error if covariance has eigenvalue <-1e-6. """
        
        min_eval = np.min(linalg.eigh(self.cov_cpl, eigvals_only=True))
        if min_eval < -1e6:
            raise ValueError('Matrix is /really/ not positive definite with min eval = '+ str(min_eval))
        elif min_eval < 0 :
            warnings.warn('Matrix is not positive definite. Smallest eigenvalue is ' + str(min_eval))
            self.cov_cpl = self.cov_cpl + (-1*min_eval + 1e-13) * np.eye(self.len_cpl)
        elif ( min_eval > 0 and min_eval < 1e-13):
            warnings.warn('Matrix is barely positive semidefinite. Smallest eigenvalue is ' + str(min_eval))
            self.cov_cpl = self.cov_cpl + 1e-13 * np.eye(self.len_cpl)

            
        
    def compute_ens_cov(self):
        """Returns ensemble covariance matrix
        
        Returns:
            cov_mat (array): ensemble covariance matrix
        """
        
        rnd = np.random.normal(size=(self.len_cpl, self.ens_size))
        
        x_err_ens = self.cov_cpl_sqrt @ rnd
        
        ens_cov = np.cov(x_err_ens)
        
        return ens_cov

    
    
    def set_ens_cov_cpl(self):
        """Computes n ensemble covariance matrices, where n = num_trials
        
        Sets Attributes:
            ens_cov_cpl (array): n ensemble covariance matrices, where n = num_trials
        """
        
        ens_cov_mats = np.empty([self.len_cpl, self.len_cpl, self.num_trials])
        
        for ii in range(self.num_trials):
            ens_cov_mats[:, :, ii] = self.compute_ens_cov()
        
        self.ens_cov_cpl = ens_cov_mats
        
        
        
    def set_vert_coord(self, xds):
        """Sets vertical coordinates in atm and ocn
        
        Args:
            xds (:obj:`xarray.Dataset`): dataset storing covariance matrices
        
        Sets Attributes:
            atm_p (array): atm pressure levels
            ocn_z (array): ocn z (depth) levels
        """
        
        self.atm_p = xds['atm_p']
        self.ocn_z = xds['ocn_z']
        
        
        
        
        
