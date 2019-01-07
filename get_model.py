
# get_model.py

import models
import robustify as rob


robfn = rob.softmean # specification of robustifier.
nun = 20 # number of units.


def get_model(mod_name, nf, nc):

    if mod_name == "shallow":

        # Note for the pure linear model, keep nobias=True.
        
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

        # Since using a non-linear model here, we set nobias=False.
        
        mod_init = models.Chain_FFWD_ReLU(dims=[nf,nun,nun,nc],
                                          robustifiers=[None,None,None],
                                          nfactors=[False,False,False],
                                          nobias=False)
        mod = models.Chain_FFWD_ReLU(dims=[nf,nun,nun,nc],
                                     robustifiers=[None,None,None],
                                     nfactors=[False,False,False],
                                     nobias=False)
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
                                          nobias=False)
        mod = models.Chain_FFWD_ReLU(dims=[nf,nun,nun,nc],
                                     robustifiers=[None,None,robfn],
                                     nfactors=[False,False,True],
                                     nobias=False)
        return (mod_init, mod)

    else:
        raise ValueError("Unknown model name.")
