import numpy as np
import configuration.config as config
import configuration.address as address 
from algorithms import *
from os import makedirs as mkdir

def gt(data_path, observation, noise, data, alpha):
    sr, grd_truth, same_range, range_gt_ave = ground_truth(output_true_test=data['output_true_test'], output_test=observation['output_test'], noise=noise['noise_test'], alpha=alpha)
    #sr, grd_truth, same_range, range_gt_ave = ground_truth2(output_true_test=data['output_true_test'], output_test=observation['output_test'], noise=noise['noise_test'], alpha=alpha)
    data_path_temp = data_path + '/base'
    mkdir(data_path_temp, exist_ok=True)
    data_path = data_path_temp + '/' + str(address.same_range['save_name']) + '.npz'
    np.savez_compressed(data_path, func_est=sr, range_ave=same_range)
    data_path = data_path_temp + '/' + str(address.ground_truth['save_name']) + '.npz'
    np.savez_compressed(data_path, func_est=grd_truth, range_ave=range_gt_ave)
    data_path = data_path_temp + '/exp_data.npz'
    np.savez_compressed(data_path, input_train=data['input_train'], input_test=data['input_test'], output_true_train=data['output_true_train'], output_true_test=data['output_true_test'], observation_train=observation['output_train'], observation_test=observation['output_test']) 
   
    #return grd_truth
    return sr

class ACIlearning():
    def __init__(self, observation, noise, Iter, alpha, trial, outlier_rate):
    #def __init__(self, observation, noise, data, alpha, method, trial, outlier_rate, observation_c, data_c):
        # Data LOADING
        alpha_range = round(alpha[1] - alpha[0], 3)
        #self.output_c = observation_c['output_test']
        #self.input_c = data_c['input_test']
        
        self.alpha = alpha
        self.Iter = Iter
        
        self.data_path_temp = 'result/text/dim=1' + '/' + str(noise['noise_type']) + '/' + str(observation['outlier_type']) + '/=' + str(outlier_rate) + '/Iter=' + str(self.Iter) + '/alpha=' + str(alpha_range) 
        self.data_path = self.data_path_temp + '/trial=' + str(trial) + '/'
        mkdir(self.data_path, exist_ok=True)
             
    def eval_ACI(self, ground_truth, coverage, func_est_final, input):
        self.coverage = coverage
        self.coverage_all = coverage
        self.grd_truth = ground_truth
        self.func_est_final = func_est_final
        self.input = input

        print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
        print("coverage")
        for i in range(10):
            print(self.coverage)
        self.ACI = {'method':'Kernel', 'variable':config.coherence, 'dict_band':config.dict_band_multi, 'save_name':'multi_kernel', 'processing':'online'}
        self.loss = {'loss':'pinball_moreau', 'gamma':0.5}
        self.range_func_est_ave, self.coverage_db = error(func_est=self.func_est_final, gt=self.grd_truth, Iter=self.Iter, method = self.ACI)
        print("-----------------------------------")
        print("error")
        print(10 * np.log10(self.coverage_db[2, 0:5]).reshape(1, -1))
        print(np.argmin(self.coverage_db[2]).reshape(1, -1))
        print(np.min(10 * np.log10(self.coverage_db[2])).reshape(1, -1))

    def save_ACI(self):
        data_path = self.data_path + '/online/' + str(self.loss['loss']) + '/\u03b3=' + str(self.loss['gamma']) 
        mkdir(data_path, exist_ok=True) 
        data_path = data_path + '/ACI.npz'
        np.savez_compressed(data_path, coverage=self.coverage, coverage_all=self.coverage_all, range_ave=self.range_func_est_ave, coverage_db=self.coverage_db, func_est=self.func_est_final, input=self.input)

class ACIlearning_lo():
    def __init__(self, observation, noise, Iter, alpha, trial, outlier_rate):
    #def __init__(self, observation, noise, data, alpha, method, trial, outlier_rate, observation_c, data_c):
        # Data LOADING
        alpha_range = round(alpha[1] - alpha[0], 3)
        #self.output_c = observation_c['output_test']
        #self.input_c = data_c['input_test']
        
        self.alpha = alpha
        self.Iter = Iter
        
        self.data_path_temp = 'result/text/dim=1' + '/' + str(noise['noise_type']) + '/' + str(observation['outlier_type']) + '/=' + str(outlier_rate) + '/Iter=' + str(self.Iter) + '/alpha=' + str(alpha_range) 
        self.data_path = self.data_path_temp + '/lo/trial=' + str(trial) + '/'
        mkdir(self.data_path, exist_ok=True)
             
    def eval_ACI(self, ground_truth, coverage, func_est_final, input):
        self.coverage = coverage
        self.coverage_all = coverage
        self.grd_truth = ground_truth
        self.func_est_final = func_est_final
        self.input = input

        print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
        print("coverage")
        for i in range(10):
            print(self.coverage)
        self.ACI = {'method':'Kernel', 'variable':config.coherence, 'dict_band':config.dict_band_multi, 'save_name':'multi_kernel', 'processing':'online'}
        self.loss = {'loss':'pinball_moreau', 'gamma':0.5}
        self.range_func_est_ave, self.coverage_db = error(func_est=self.func_est_final, gt=self.grd_truth, Iter=self.Iter, method = self.ACI)
        print("-----------------------------------")
        print("error")
        print(10 * np.log10(self.coverage_db[2, 0:5]).reshape(1, -1))
        print(np.argmin(self.coverage_db[2]).reshape(1, -1))
        print(np.min(10 * np.log10(self.coverage_db[2])).reshape(1, -1))

    def save_ACI(self):
        data_path = self.data_path + '/online/' + str(self.loss['loss']) + '/\u03b3=' + str(self.loss['gamma']) 
        mkdir(data_path, exist_ok=True) 
        data_path = data_path + '/ACI.npz'
        np.savez_compressed(data_path, coverage=self.coverage, coverage_all=self.coverage_all, range_ave=self.range_func_est_ave, coverage_db=self.coverage_db, func_est=self.func_est_final, input=self.input)


class ACIlearning_hi():
    def __init__(self, observation, noise, Iter, alpha, trial, outlier_rate):
    #def __init__(self, observation, noise, data, alpha, method, trial, outlier_rate, observation_c, data_c):
        # Data LOADING
        alpha_range = round(alpha[1] - alpha[0], 3)
        #self.output_c = observation_c['output_test']
        #self.input_c = data_c['input_test']
        
        self.alpha = alpha
        self.Iter = Iter
        
        self.data_path_temp = 'result/text/dim=1' + '/' + str(noise['noise_type']) + '/' + str(observation['outlier_type']) + '/=' + str(outlier_rate) + '/Iter=' + str(self.Iter) + '/alpha=' + str(alpha_range) 
        self.data_path = self.data_path_temp + '/hi/trial=' + str(trial) + '/'
        mkdir(self.data_path, exist_ok=True)
             
    def eval_ACI(self, ground_truth, coverage, func_est_final, input):
        self.coverage = coverage
        self.coverage_all = coverage
        self.grd_truth = ground_truth
        self.func_est_final = func_est_final
        self.input = input

        print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
        print("coverage")
        for i in range(10):
            print(self.coverage)
        self.ACI = {'method':'Kernel', 'variable':config.coherence, 'dict_band':config.dict_band_multi, 'save_name':'multi_kernel', 'processing':'online'}
        self.loss = {'loss':'pinball_moreau', 'gamma':0.5}
        self.range_func_est_ave, self.coverage_db = error(func_est=self.func_est_final, gt=self.grd_truth, Iter=self.Iter, method = self.ACI)
        print("-----------------------------------")
        print("error")
        print(10 * np.log10(self.coverage_db[2, 0:5]).reshape(1, -1))
        print(np.argmin(self.coverage_db[2]).reshape(1, -1))
        print(np.min(10 * np.log10(self.coverage_db[2])).reshape(1, -1))

    def save_ACI(self):
        data_path = self.data_path + '/online/' + str(self.loss['loss']) + '/\u03b3=' + str(self.loss['gamma']) 
        mkdir(data_path, exist_ok=True) 
        data_path = data_path + '/ACI.npz'
        np.savez_compressed(data_path, coverage=self.coverage, coverage_all=self.coverage_all, range_ave=self.range_func_est_ave, coverage_db=self.coverage_db, func_est=self.func_est_final, input=self.input)


class ACIlearning_hal():
    def __init__(self, observation, noise, Iter, alpha, trial, outlier_rate):
    #def __init__(self, observation, noise, data, alpha, method, trial, outlier_rate, observation_c, data_c):
        # Data LOADING
        alpha_range = round(alpha[1] - alpha[0], 3)
        #self.output_c = observation_c['output_test']
        #self.input_c = data_c['input_test']
        
        self.alpha = alpha
        self.Iter = Iter
        
        self.data_path_temp = 'result/text/dim=1' + '/' + str(noise['noise_type']) + '/' + str(observation['outlier_type']) + '/=' + str(outlier_rate) + '/Iter=' + str(self.Iter) + '/alpha=' + str(alpha_range) 
        self.data_path = self.data_path_temp + '/hal/trial=' + str(trial) + '/'
        mkdir(self.data_path, exist_ok=True)
             
    def eval_ACI(self, ground_truth, coverage, func_est_final, input):
        self.coverage = coverage
        self.coverage_all = coverage
        self.grd_truth = ground_truth
        self.func_est_final = func_est_final
        self.input = input

        print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
        print("coverage")
        for i in range(10):
            print(self.coverage)
        self.ACI = {'method':'Kernel', 'variable':config.coherence, 'dict_band':config.dict_band_multi, 'save_name':'multi_kernel', 'processing':'online'}
        self.loss = {'loss':'pinball_moreau', 'gamma':0.5}
        self.range_func_est_ave, self.coverage_db = error(func_est=self.func_est_final, gt=self.grd_truth, Iter=self.Iter, method = self.ACI)
        print("-----------------------------------")
        print("error")
        print(10 * np.log10(self.coverage_db[2, 0:5]).reshape(1, -1))
        print(np.argmin(self.coverage_db[2]).reshape(1, -1))
        print(np.min(10 * np.log10(self.coverage_db[2])).reshape(1, -1))

    def save_ACI(self):
        data_path = self.data_path + '/online/' + str(self.loss['loss']) + '/\u03b3=' + str(self.loss['gamma']) 
        mkdir(data_path, exist_ok=True) 
        data_path = data_path + '/ACI.npz'
        np.savez_compressed(data_path, coverage=self.coverage, coverage_all=self.coverage_all, range_ave=self.range_func_est_ave, coverage_db=self.coverage_db, func_est=self.func_est_final, input=self.input)