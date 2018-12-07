
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
    Catoni and Giulini (2017) key insights applied
    to a one-dimensional setting, very handy.
    
    Here "x" is assumed to be an ndarray with
    no structure, just (k,) shape.
    '''

    est_mean = np.mean(x)
    est_var = max(np.var(x), 0.001)
    est_sd = math.sqrt(est_var)

    # Treat data as z-scores, thus no need for "safe" bounds.
    Tbound = 1.0 
    vbound = Tbound
    lam = math.sqrt(2*math.log(1/config.CONF_DELTA)/(x.size*vbound))
    beta = math.sqrt(
        2*Tbound*math.log(1/config.CONF_DELTA)/vbound
    )
    xhat = hlp.est_robust(x=(x-est_mean)/est_sd, lam=lam, beta=beta)
    xhat *= est_sd
    xhat += est_mean

    return xhat



