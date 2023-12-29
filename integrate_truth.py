from configuration import config
from configuration import address
from configuration import graph_config as grp
import numpy as np
from os import makedirs as mkdir

import scipy as sp
import scipy.stats as st

trial = config.trial
cov_temp = np.zeros(trial)
cov_db_temp = np.zeros([trial, 3])
cov_db_interval = np.zeros([3, 2])

alpha_rel = 0.95
deg_free = trial - 1

string = "OCQKR, OCQKR0, OCQKR100"
for gamma in config.gamma:
    for OCQ in string.split(', '):
        for method in config.methods:
            for i in range(config.trial):
                # get path
                data_path_detail = 'truth/linear_expansion/sparse/outlier_rate=0.04/Iter=3000/alpha=0.95/trial=' + str(i+1) + '/online/pinball_moreau/γ=' + str(gamma) + '/' + str(OCQ) + '/' + str(method) + '.npz'
                result = np.load(data_path_detail)
                
                if i == 0:
                    coverage_db = result['coverage_db'] / trial
                    range_ave = result['range_ave'] / trial

                else:        
                    coverage_db += result['coverage_db'] / trial
                    range_ave += result['range_ave'] / trial
                
                cov_db_temp[i] = result['coverage_db'][:, -1].reshape(-1) 
                
            #cov_scale = np.sqrt(st.tvar(cov_temp) / trial)
            #if OCQ == 'OCQKR':
                #cov_path = 'result/text/dim=1/linear_expansion/sparse/outlier_rate=0.04/Iter=1000/alpha=0.95/online/pinball_moreau/γ=' + str(gamma) + '/CQR/' + str(method) + '.npz'
            #elif OCQ == 'OCQKR0':
                #cov_path = 'result/text/dim=1/linear_expansion/sparse/outlier_rate=0.04/Iter=1000/alpha=0.95/online/pinball_moreau/γ=' + str(gamma) + '/CQR0/' + str(method) + '.npz'
            #elif OCQ == 'OCQKR100':
                #cov_path = 'result/text/dim=1/linear_expansion/sparse/outlier_rate=0.04/Iter=1000/alpha=0.95/online/pinball_moreau/γ=' + str(gamma) + '/CQR100/' + str(method) + '.npz'
            #cov = np.load(cov_path)
            #cov_ave = (cov['coverage'][1] - cov['coverage'][0]).reshape(-1)

            #for j in range(3):
                #cov_ave_db = st.tmean(cov_db_temp[:, j])
                #cov_scale_db = np.sqrt(st.tvar(cov_db_temp[:, j]) / trial)
                #cov_db_interval[j] = st.t.interval(alpha_rel, deg_free, loc=cov_ave_db, scale=cov_scale_db)
            
            #cov_interval = st.t.interval(alpha_rel, deg_free, loc=cov_ave, scale=cov_scale)
            # save
            save_path = 'truth/linear_expansion/sparse/outlier_rate=0.04/Iter=3000/alpha=0.95/online/pinball_moreau/γ=' + str(gamma) + '/' + str(OCQ) + '/' + str(method) + '.npz'
            mk_path = 'truth/linear_expansion/sparse/outlier_rate=0.04/Iter=3000/alpha=0.95/online/pinball_moreau/γ=' + str(gamma) + '/' + str(OCQ)
            print(save_path)
            mkdir(mk_path, exist_ok=True)
            #np.savez_compressed(save_path, coverage_interval=cov_interval, coverage_db=10 * np.log10(coverage_db), coverage_db_interval=(10 * np.log10(cov_db_interval)), range_ave=range_ave)
            np.savez_compressed(save_path, coverage_db=10 * np.log10(coverage_db), coverage_db_interval=(10 * np.log10(cov_db_interval)), range_ave=range_ave)

#if __name__ == '__main__':
    #for alpha_range_temp in range(config.limit):
        #alpha_range = np.round(0.95 - (alpha_range_temp * 0.05), 3)
    
        #data_path =  'result/text/dim=' + str(config.input_dim) + '/' + str(config.noise_type) + '/' + str(config.outlier_type) + '/Iter=' + str(config.Iter) + '/alpha=' + str(alpha_range)

        #for _, method in enumerate(config.methods):
            #data_integrate(data_path=data_path, method=method, loss=config.loss, gamma=config.gamma_default, trial=config.trial, input_te=input_te)