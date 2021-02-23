
## Various helper functions.

import os
import math
import numpy as np
import scipy.special as spec

import config


def makedir_safe(dirname):
    '''
    A simple utility for making new directories
    after checking that they do not exist.
    '''
    if not os.path.exists(dirname):
        os.makedirs(dirname)
        

def psi_fn(u):
    '''
    Influence function of Catoni and Giulini (2017).
    '''
    return np.where((np.abs(u) > math.sqrt(2)),\
                    (np.sign(u)*2*math.sqrt(2)/3),\
                    (u-u**3/6))


