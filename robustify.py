
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
    
    NOTE: "calib" refers to a sort of calibration
    that we do when setting the v-bound.
    No additional factors here.
    
    Here "x" is assumed to be an ndarray with
    no structure, just (k,) shape.
    '''

    # A couple statistics.
    est_mean = np.mean(x)
    est_var = max(np.var(x), 0.001)
    
    # Checks.
    calib_shift = (abs(est_mean) / est_var) > config.CALIB_THRES
    calib_scale = not calib_shift

    # If the variance is not much smaller than the absolute
    # position, then temporarily downscale, do estimation, and
    # then return to original scale.
    if calib_scale:
        Tbound = (est_var + est_mean**2) / config.CALIB_S0
        Tbound_safe = max(Tbound, config.T_MIN)
        vbound_safe = Tbound_safe
        lam = math.sqrt(2*math.log(1/config.CONF_DELTA)/(x.size*vbound_safe))
        beta = math.sqrt(
            2*Tbound_safe*math.log(1/config.CONF_DELTA)/vbound_safe
        )
        xhat = hlp.est_robust(x=x/config.CALIB_S0,
                              lam=lam, beta=beta) * config.CALIB_S0

    # If the variance is much smaller than the position, then a
    # simple temporary shift will be sufficient to combat bias.
    elif calib_shift:
        Tbound = est_var + est_mean**2
        Tbound_safe = max(Tbound, config.T_MIN)
        vbound_safe = Tbound_safe
        lam = math.sqrt(2*math.log(1/config.CONF_DELTA)/(x.size*vbound_safe))
        beta = math.sqrt(
            2*Tbound_safe*math.log(1/config.CONF_DELTA)/vbound_safe
        )
        xhat = hlp.est_robust(x=(x-est_mean),
                              lam=lam, beta=beta) + est_mean

    else:
        Tbound = est_var + est_mean**2
        Tbound_safe = max(Tbound, config.T_MIN)
        vbound_safe = Tbound_safe
        lam = math.sqrt(2*math.log(1/config.CONF_DELTA)/(x.size*vbound_safe))
        beta = math.sqrt(
            2*Tbound_safe*math.log(1/config.CONF_DELTA)/vbound_safe
        )
        xhat = hlp.est_robust(x=x, lam=lam, beta=beta)

    return xhat





