
# get_model.py

import models
import robustify as rob


robfn = rob.softmean # specification of robustifier.
nun = 20 # number of units.


def get_model(mod_name, nf, nc):

    if mod_name == "shallow":
        mod_init = models.Chain_FFWD_ReLU(dims=[nf,nc],
                                          robustifiers=[None],
                                          nfactors=[False],
                                          nobias=True)
        mod = models.Chain_FFWD_ReLU(dims=[nf,nc],
                                     robustifiers=[None],
                                     nfactors=[False],
                                     nobias=True)
        return (mod_init, mod)

    elif mod_name == "deep":
        mod_init = models.Chain_FFWD_ReLU(dims=[nf,nun,nun,nc],
                                          robustifiers=[None,None,None],
                                          nfactors=[False,False,False],
                                          nobias=True)
        mod = models.Chain_FFWD_ReLU(dims=[nf,nun,nun,nc],
                                     robustifiers=[None,None,None],
                                     nfactors=[False,False,False],
                                     nobias=True)
        return (mod_init, mod)

    elif mod_name == "shallow-rob":
        mod_init = models.Chain_FFWD_ReLU(dims=[nf,nc],
                                          robustifiers=[robfn],
                                          nfactors=[True],
                                          nobias=True)
        mod = models.Chain_FFWD_ReLU(dims=[nf,nc],
                                     robustifiers=[robfn],
                                     nfactors=[True],
                                     nobias=True)
        return (mod_init, mod)

    elif mod_name == "deep-rob":
        mod_init = models.Chain_FFWD_ReLU(dims=[nf,nun,nun,nc],
                                          robustifiers=[None,None,robfn],
                                          nfactors=[False,False,True],
                                          nobias=True)
        mod = models.Chain_FFWD_ReLU(dims=[nf,nun,nun,nc],
                                     robustifiers=[None,None,robfn],
                                     nfactors=[False,False,True],
                                     nobias=True)
        return (mod_init, mod)

    else:
        raise ValueError("Unknown model name.")
