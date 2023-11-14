import graph
import data as dt
import optimize
from integrate import data_integrate as integrate
from ACI import runACI
import csv

import configuration.config as config
import configuration.graph_config as graph_config
import configuration.address as address

import numpy as np
# data 

alpha_all = config.alpha_all[config.start:config.limit]
alpha_range = config.alpha_range

if config.onlyACIflag == 'on':
    for noise_type in config.noise_type_all:
        for outlier_type in config.outlier_type_all:
            for outlier_rate in config.outlier_rate:
                for i in range(config.trial):
                    data_path = 'exp_data/' + 'dim=' + str(config.input_dim) + '/' + str(noise_type) + '/' + str(outlier_type) + '/outlier_rate=' + str(outlier_rate) + '/Iter=' + str(config.Iter) + '/trial=' + str(i + 1)
                    dt.dt(data_path=data_path, Iter=config.Iter, input_dim=config.input_dim, noise_type=noise_type, outlier_type=outlier_type, outlier_rate=outlier_rate)

for noise_type in config.noise_types:
    for outlier_type in config.outlier_types:
        for outlier_rate in config.outlier_rate:
            with open('log2.txt', 'w') as f:
                f.write('noise_type : ' + str(noise_type))
                f.write('\noutlier_type : ' + str(outlier_type))
                f.write('\noutlier_rate : ' + str(outlier_rate))
                f.write('\n---------------------------------------------')

            
            for index_alpha, alpha in enumerate(alpha_all):
                with open('log2.txt', 'a') as f:
                    f.write('\n---------------------------------------------')
                    f.write('\n' +  str(index_alpha + 1) + ' / ' + str(len(alpha_all)) + ' : ' + str(alpha))
                    f.write('\n---------------------------------------------')

                for i in range(config.trial):
                    data_path = 'exp_data/' + 'dim=' + str(config.input_dim) + '/' + str(noise_type) + '/' + str(outlier_type) + '/outlier_rate=' + str(outlier_rate) + '/Iter=' + str(config.Iter) + '/trial=' + str(i+1) + '/' 
                    observation = np.load(data_path + 'outlier.npz')
                    noise = np.load(data_path + 'noise.npz')
                    data = np.load(data_path + 'data.npz')
                                
        runACI(output=observation['output_test'], input=data['input_test'], alpha=alpha_all, alpha_range=alpha_range, step=0.005, tinit=1900, splitSize=0.2)

        with open('log2.txt', 'a') as f:
            f.write('\n---------------------------------------------')
            f.write('END')