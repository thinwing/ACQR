import graph
#import data as dt
import data2 as dt
import optimize
from integrate import data_integrate as integrate

import datetime
import configuration.config as config
import configuration.graph_config as graph_config
import configuration.address as address

from os import makedirs as mkdir
import numpy as np

for noise_type in config.noise_type_all:
    for outlier_type in config.outlier_type_all:
        for outlier_rate in config.outlier_rate:
            for i in range(5, config.trial):
                data_path = 'exp_data/' + 'dim=' + str(config.input_dim) + '/' + str(noise_type) + '/' + str(outlier_type) + '/outlier_rate=' + str(outlier_rate) + '/Iter=' + str(config.Iter) + '/trial=' + str(i + 1)
                dt.dt2(data_path=data_path, Iter=config.Iter, input_dim=config.input_dim, noise_type=noise_type, outlier_type=outlier_type, outlier_rate=outlier_rate)