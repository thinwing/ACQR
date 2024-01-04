import graph
#import data as dt
import data3 as dt
import optimize
from integrate import data_integrate as integrate

import datetime
import configuration.config as config
import configuration.graph_config as graph_config
import configuration.address as address

from os import makedirs as mkdir
import numpy as np

noise_type = 'linear_expansion'
outlier_type = 'sparse'
outlier_rate = 0.05
for i in range(config.trial):
    data_path = 'exp_data/' + 'dim=' + str(config.input_dim) + '/' + str(noise_type) + '/' + str(outlier_type) + '/outlier_rate=' + str(outlier_rate) + '/Iter=' + str(config.Iter) + '/trial=' + str(i + 1)
    #dt.dtlo(data_path=data_path, Iter=config.Iter, input_dim=config.input_dim, noise_type=noise_type, outlier_type=outlier_type, outlier_rate=outlier_rate)
    #dt.dthi(data_path=data_path, Iter=config.Iter, input_dim=config.input_dim, noise_type=noise_type, outlier_type=outlier_type, outlier_rate=outlier_rate)
    dt.dthal(data_path=data_path, Iter=config.Iter, input_dim=config.input_dim, noise_type=noise_type, outlier_type=outlier_type, outlier_rate=outlier_rate)