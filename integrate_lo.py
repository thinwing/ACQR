import graph
#import data as dt
import data2 as dt
import optimize
#from integrate import data_integrate_hi as integrate
#from integrate import data_integrate_CQRhi as integrateCQR

#from integrate import data_integrate_lo as integrate
#from integrate import data_integrate_CQRlo as integrateCQR

from integrate import data_integrate_hal as integrate
from integrate import data_integrate_CQRhal as integrateCQR


import datetime
import configuration.config as config
import configuration.graph_config as graph_config
import configuration.address as address

from os import makedirs as mkdir
import numpy as np

for method in config.methods:
    for loss_temp in config.losses:
        for gamma in config.gamma:
            data_path_temp = 'result/text/dim=1/linear_expansion/sparse/outlier_rate=0.05/Iter=3000/alpha=0.95'
            integrate(data_path=data_path_temp, method=str(method), loss=loss_temp, gamma=gamma, trial=config.trial)

for method in config.methods:
    for loss_temp in config.losses:
        for gamma in config.gamma:
            data_path_temp = 'result/text/dim=1/linear_expansion/sparse/outlier_rate=0.05/Iter=1000/alpha=0.95'
            integrateCQR(data_path=data_path_temp, method=str(method), loss=loss_temp, gamma=gamma, trial=config.trial)