
## Various helper functions.

import os
import numpy as np
import math
import config


def makedir_safe(dirname):
    '''
    A simple utility for making new directories
    after checking that they do not exist.
    '''
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def normCDF_1D(u):
    '''
    1-dim version of Normal CDF.
    See p.934 of Abramowitz and Stegun (eqn 26.2.29).
    '''
    return (1 + math.erf(u/math.sqrt(2))) / 2

normCDF = np.vectorize(normCDF_1D) # Vectorized Normal CDF.


def correction(m, sigma):
    '''
    The additive correction term in the Catoni-type estimator.
    '''
    # Some quantities to save.
    vm = (math.sqrt(2) - m) / sigma
    vp = (math.sqrt(2) + m) / sigma
    Fm = normCDF(-vm)
    Fp = normCDF(-vp)
    em = np.exp(-vm**2/2)
    ep = np.exp(-vp**2/2)

    # Broken up into five terms.
    t1 = 2*math.sqrt(2) * (Fm - Fp) / 3

    t2 = (-1) * (m-m**3/6) * (Fm + Fp)

    t3 = sigma * (1-m**2/2) * (ep - em) / math.sqrt(2*math.pi)

    t4 = m * sigma**2 * (Fp+Fm+(vp*ep + vm*em)/math.sqrt(2*math.pi)) / 2

    t5 = sigma**3 * ((2+vm**2)*em-(2+vp**2)*ep) / (6*math.sqrt(2*math.pi))

    # Just sum them up and return.
    return t1 + t2 + t3 + t4 + t5


def est_robust(x, lam, beta):
    '''
    New Catoni-type estimator.
    Assumes that x is a vector of shape (n,k), where
    rows correspond to distinct observations. The
    parameters lam and beta are assumed to have 
    shape (k,) or (), can be just scalars.
    '''

    # Shape checks.
    n,k = x.shape
    if len(x.shape) < 2:
        raise ValueError("Shapes are not as expected.")

    # Main computations have no issues with under/overflow.
    comps = x * (1 - (lam*x)**2/(2*beta)) - (lam**2)*(x**3)/6
    
    # Make sure things are numerically stable for corrections.
    # note: beta is safe as-is.
    lam_safe = np.clip(a=lam, a_min=config.LAM_MIN, a_max=None)

    # Final computations based on safe values.
    corr = correction(
        m=lam_safe*x,
        sigma=np.where((lam_safe*np.abs(x) < config.SIGMA_MIN),
                       config.SIGMA_MIN,
                       lam_safe*np.abs(x))/math.sqrt(beta)
    ) / lam_safe
    return np.mean(comps, axis=0) + np.mean(corr, axis=0)
