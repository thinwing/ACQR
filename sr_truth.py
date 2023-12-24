import numpy as np
from os import makedirs as mkdir
import sys
sys.path.append('../')
from configuration import config
from integrate import get_path
from integrate import get_path_CQR
import configuration.address as address 
from algorithms import *

#for outlier_rate in config.outlier_rate:
outlier_rate = 0.04
for i in range(config.trial):
    #truth
    data_path = 'result/text/dim=1/linear_expansion/sparse/outlier_rate=' + str(outlier_rate) + '/Iter=' + str(config.Iter) + '/alpha=0.95/trial=' + str(i+1)
    sr_path = data_path + '/base/same_range.npz'
    ground_result = np.load(sr_path)
    grd_truth = ground_result['func_est']
    print(np.size(grd_truth))
    ACI = {'method':'Kernel', 'variable':config.coherence, 'dict_band':config.dict_band_multi, 'save_name':'multi_kernel', 'processing':'online'}
    
    #ACI
    #data_path = 'result/text/dim=1/linear_expansion/sparse/=' + str(outlier_rate) + '/Iter=' + str(config.Iter) + '/alpha=0.95/trial=' + str(i+1)
    #func_path = data_path + '/online/pinball_moreau/\u03b3=0.5/ACI.npz'
    #func_result = np.load(func_path)
    #func_est = func_result['func_est']

    #range_func_est_ave, coverage_db = error(func_est=func_est, gt=grd_truth, Iter=config.Iter, method = ACI)
    #print(np.size(func_est))
    #save_path = 'truth/linear_expansion/sparse/outlier_rate=' + str(outlier_rate) + '/Iter=' + str(config.Iter) + '/alpha=0.95/trial=' + str(i+1) + '/online/pinball_moreau/\u03b3=0.5'
    #mkdir(save_path, exist_ok=True)
    #save_path += '/'
    #np.savez_compressed(save_path + 'ACI.npz', range_ave=range_func_est_ave, coverage_db=coverage_db, func_est=func_est)
    
    for gamma in config.gamma:
        for method in config.methods:
            data_path = 'result/text/dim=1/linear_expansion/sparse/outlier_rate=' + str(outlier_rate) + '/Iter=' + str(config.Iter) + '/alpha=0.95/trial=' + str(i+1)
            func_path = data_path + '/online/pinball_moreau/' + '\u03b3=' + str(gamma) + '/' + str(method) + '.npz'
            func_result = np.load(func_path)
            func_est = func_result['func_est']
            print(np.size(func_est))
            print('err')
            print(func_path)
            print(np.size(grd_truth))
            range_func_est_ave, coverage_db = error(func_est=func_est, gt=grd_truth, Iter=config.Iter, method = ACI)

            save_path = 'truth/linear_expansion/sparse/outlier_rate=' + str(outlier_rate) + '/Iter=' + str(config.Iter) + '/alpha=0.95/trial=' + str(i+1) + '/online/pinball_moreau/' + '\u03b3=' + str(gamma) + '/OQKR'
            
            mkdir(save_path, exist_ok=True)
            save_path += '/'
            np.savez_compressed(save_path + str(method) + '.npz', range_ave=range_func_est_ave, coverage_db=coverage_db, func_est=func_est)
        #OCQKR
        for method in config.methods:
            data_path = 'result/text/dim=1/linear_expansion/sparse/outlier_rate=' + str(outlier_rate) + '/Iter=' + str(config.Iter_CQR) + '/alpha=0.95/trial=' + str(i+1)
            sr_path = data_path + '/base/CQR/same_range.npz'
            ground_result = np.load(sr_path)
            grd_truth_CQR = ground_result['func_est']
            
            func_path = data_path + '/online/pinball_moreau/' + '\u03b3=' + str(gamma) + '/CQR/' + str(method) + '.npz'
            func_result = np.load(func_path)
            func_est = func_result['func_est']
            range_func_est_ave, coverage_db = error(func_est=func_est, gt=grd_truth_CQR, Iter=config.Iter_CQR, method = ACI)

            save_path = 'truth/linear_expansion/sparse/outlier_rate=' + str(outlier_rate) + '/Iter=' + str(config.Iter) + '/alpha=0.95/trial=' + str(i+1) + '/online/pinball_moreau/' + '\u03b3=' + str(gamma) + '/OCQKR'
            
            mkdir(save_path, exist_ok=True)
            save_path += '/'
            np.savez_compressed(save_path + str(method) + '.npz', range_ave=range_func_est_ave, coverage_db=coverage_db, func_est=func_est)