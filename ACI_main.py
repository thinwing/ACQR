import graph
import data as dt
import optimize_ACI 
import datetime
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
                for i in range(25, config.trial):
                    data_path = 'exp_data/' + 'dim=' + str(config.input_dim) + '/' + str(noise_type) + '/' + str(outlier_type) + '/outlier_rate=' + str(outlier_rate) + '/Iter=' + str(config.Iter) + '/trial=' + str(i + 1)
                    dt.dt(data_path=data_path, Iter=config.Iter, input_dim=config.input_dim, noise_type=noise_type, outlier_type=outlier_type, outlier_rate=outlier_rate)

with open('log2.txt', 'a') as f:
    f.write('start')
    f.write('\n---------------------------------------------')
for noise_type in config.noise_types:
    for outlier_type in config.outlier_types:
        #for outlier_rate in config.outlier_rate:
        outlier_rate = 0.05
        with open('log2.txt', 'a') as f:
            f.write('\nnoise_type : ' + str(noise_type))
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
                ACI_data = runACI(output=observation['output_test'], input=data['input_test'], alpha=alpha_all, alpha_range=alpha_range, step=0.005, tinit=1000, splitSize=0.5)
                coverage = ACI_data[0]
                input_ACI = ACI_data[1]
                func_est_final = ACI_data[2]
                now = datetime.datetime.now()
                #data_path_temp = data_path
                #data_path = 'truth/' + str(noise_type) + '/' + str(outlier_type) + '/outlier_rate=' + str(outlier_rate)  +'/Iter=' + str(config.Iter) + '/trial=' + str(i+1) + '/' 
                #true = np.load(data_path + 'grd_truth.npz')
                #grd_truth = true['arr_0']
                #data_path = data_path_temp
                #sr_path = 'result/text/dim=1/linear_expansion/sparse/outlier_rate=' + str(outlier_rate) + '/Iter=' + str(config.Iter) + '/alpha=0.95/trial=' + str(i+1) + '/base/same_range.npz'
                #ground_result = np.load(sr_path)
                #grd_truth = ground_result['func_est']
                
                learn = optimize_ACI.ACIlearning(observation=observation, noise=noise, Iter=config.Iter, alpha=alpha, trial=i+1, outlier_rate=outlier_rate)
                grd_truth = optimize_ACI.gt(data_path=learn.data_path, observation=observation, noise=noise, data=data, alpha=alpha)
                learn.eval_ACI(ground_truth=grd_truth, coverage=coverage, func_est_final=func_est_final, input=input_ACI)
                learn.save_ACI()
                with open('log2.txt', 'a') as f:
                    f.write('\n' + '\t' + str(i + 1) + ' / ' + str(config.trial) + ' : ' + str(now))
                    f.write('\n' + '\t\tCoverage rate = ' + str(coverage))
                                
        with open('log2.txt', 'a') as f:
            f.write('\n---------------------------------------------')
            f.write('END')