
# robustify.py

import math
import numpy as np

import config
import helpers as hlp

def softmean(x):
    '''
    Here "x" is assumed to have shape (n,k), where
    the rows (the first index) correspond to the
    observations.
    '''
    
    n, k = x.shape
    est_mean = np.mean(x, axis=0)
    est_var = np.clip(a=np.var(x, axis=0), a_min=0.001, a_max=None)
    est_sd = np.sqrt(est_var)
    s = math.sqrt(n / (2*math.log(1/config.CONF_DELTA)))
    xhat = np.mean(hlp.psi_fn(u=(x-est_mean)/(est_sd*s)), axis=0)
    xhat *= s*est_sd # back to original scale
    xhat += est_mean # back to original location
    
    return xhat


