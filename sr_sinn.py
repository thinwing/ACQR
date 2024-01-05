from configuration import config
from configuration import address
from configuration import graph_config as grp
import numpy as np
from os import makedirs as mkdir
from algorithms import *
from range_get import range_get
import scipy as sp
import scipy.stats as st

for noise_type in config.noise_types:
    for outlier_type in config.outlier_types:
        #for outlier_rate in config.outlier_rate:
        outlier_rate = 0.05
        for method in config.methods:
            for i in range(1, config.trial):
                data_path = 'exp_data/' + 'dim=' + str(config.input_dim) + '/' + str(noise_type) + '/' + str(outlier_type) + '/outlier_rate=' + str(outlier_rate) + '/Iter=' + str(config.Iter) + '/trial=' + str(i+1) + '/'
                observation = np.load(data_path + 'outlier.npz')
                noise = np.load(data_path + 'noise.npz')
                data = np.load(data_path + 'data.npz')
                grd_path = 'result/text/' + 'dim=' + str(config.input_dim) + '/' + str(noise_type) + '/' + str(outlier_type) + '/outlier_rate=' + str(outlier_rate) + '/Iter=' + str(config.Iter) + '/alpha=0.95/trial=' + str(i+1) + '/base'+ '/same_range.npz'    
                grd = np.load(grd_path)
                grd_truth = grd['func_est']
                for gamma in config.gamma:
                    func_path = 'result/text/' + 'dim=' + str(config.input_dim) + '/' + str(noise_type) + '/' + str(outlier_type) + '/outlier_rate=' + str(outlier_rate) + '/Iter=' + str(config.Iter_CQR) + '/alpha=0.95/trial=' + str(i+1) +  '/online/pinball_moreau/\u03b3=' + str(gamma)  + '/CQR'+ '/' + str(method) + '.npz'
                    func = np.load(func_path)
                    func_est_all = func['func_est_all']
                    func_est = func['func_est']
                    coverage = func['coverage']
                    coverage_all = func['coverage_all']
                    input_test = data['input_test']
                    input_te = func['input_te']
                    multi = {'method':'Kernel', 'variable':config.coherence, 'dict_band':config.dict_band_multi, 'save_name':'multi_kernel', 'processing':'online'}
                    range_func_est_ave, coverage_db = error(func_est=func_est_all, gt=grd_truth, Iter=config.Iter, method=multi)
                    save_path = 'result/text/' + 'dim=' + str(config.input_dim) + '/' + str(noise_type) + '/' + str(outlier_type) + '/outlier_rate=' + str(outlier_rate) + '/Iter=' + str(config.Iter_CQR) + '/alpha=0.95/trial=' + str(i+1) +  '/online/pinball_moreau/\u03b3=' + str(gamma)  + '/CQR' 
                    mkdir(save_path, exist_ok=True) 
                    save_path = save_path + '/' + str(method) + '.npz'
                    print(save_path)
                    np.savez_compressed(save_path, coverage=coverage, coverage_all=coverage_all, range_ave=range_func_est_ave, coverage_db=coverage_db, func_est=func_est, func_est_all=func_est_all, input_te = input_te, input_test = input_test)