
# get_model.py

import models
import robustify as rob


def get_model(mod_name, nf, nc, paras=None):


    if mod_name == "deep":

        nun = paras["num_units"]
        mod = models.Chain_Class_H2_ReLU_Robust(
            out_l0=nf,
            out_l1=nun,
            out_l2=nun,
            out_l3=nc,
            robustifiers=[None,None,None],
            nfactors=[False,False,False]
        )
        
        return mod

    elif mod_name == "deep-rob":

        nun = paras["num_units"]
        robfn = paras["robfn"]
        mod = models.Chain_Class_H2_ReLU_Robust(
            out_l0=nf,
            out_l1=nun,
            out_l2=nun,
            out_l3=nc,
            robustifiers=[robfn,None,None],
            nfactors=[False,False,False]
        )
        
        return mod

    else:
        raise ValueError("Unknown model name.")
