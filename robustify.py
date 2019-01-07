
# robustify.py
# A collection of functions with a standardized form for use
# within our prototyping framework. The core computations are
# implemented within "helpers".

import math
import numpy as np

import config
import helpers as hlp


def softmean(x, paras=None):
    '''
    Our simplification and modification of the
    key ideas of Catoni and Giulini (2017) to
    make for a readily computable robust mean.
    
    Here "x" is assumed to have shape (n,k), where
    the rows (the first index) correspond to the
    observations.
    '''

    n, k = x.shape

    if paras is None:
        est_mean = np.mean(x, axis=0)
        est_var = np.clip(a=np.var(x, axis=0), a_min=0.001, a_max=None)
        est_sd = np.sqrt(est_var)
    else:
        est_mean = paras["mean"]
        est_var = paras["var"]
        est_sd = np.sqrt(est_var)

    # Treat data as z-scores, thus no need for "safe" bounds.
    Tbound = 1.0 
    vbound = Tbound
    lam = math.sqrt(2*math.log(1/config.CONF_DELTA)/(n*vbound))
    beta = math.sqrt(
        2*Tbound*math.log(1/config.CONF_DELTA)/vbound
    )
    xhat = hlp.est_robust(x=(x-est_mean)/est_sd, lam=lam, beta=beta)
    xhat *= est_sd
    xhat += est_mean

    return xhat
    


