import numpy as np
import configuration.config as config
import configuration.address as address 
from algorithms import *
from os import makedirs as mkdir
   
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
             
    def eval_ACI(self, ground_truth, coverage, func_est_final):
        self.coverage = coverage
        self.coverage_all = coverage
        self.grd_truth = ground_truth
        self.func_est_final = func_est_final

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
        np.savez_compressed(data_path, coverage=self.coverage, coverage_all=self.coverage_all, range_ave=self.range_func_est_ave, coverage_db=self.coverage_db, func_est=self.func_est_final)

    #def CQR(self, data_c, observation_c):
        #self.output_c = observation_c['output_test']
        #self.input_c = data_c['input_test']
        #ol_c = eval(self.method['method'])(input=self.input_c, dict_band=self.method['dict_band'])        
        #ol_c.dict_define(self.method['variable'])
        #self.calib_vector = ol_c.kernel_vector(self.input_c)

        #self.func_calib = np.zeros([len(self.alpha), 1, len(self.output_train)])
        #for i in range(self.Iter):        
            # Pinball Moreau
        #for a in range(len(self.alpha)):
        #    self.func_calib[a,:,:] = np.dot(self.kernel_weight[a].T, self.calib_vector)
        #self.func_low_c = self.func_calib[0].T
        #self.func_high_c = self.func_calib[1].T
        #self.scores_c = np.maximum(self.output_c - self.func_high_c.reshape(-1, 1), self.func_low_c.reshape(-1, 1) - self.output_c)
        #self.confQuantAdapt_c = np.percentile(self.scores_c, config.alpha_range * 100)
        #self.X_c = np.full([len(self.scores_c), 1], self.confQuantAdapt_c)
        #self.higher_c = self.func_high_c.reshape(-1, 1) + self.X_c.reshape(-1, 1)
        #self.lower_c = self.func_low_c.reshape(-1, 1) - self.X_c.reshape(-1, 1)
        #self.coverage_h_c = np.where((self.higher_c - self.output_c > 0), 1, 0)
        #self.coverage_l_c = np.where((self.lower_c - self.output_c > 0), 1, 0)
        #XX = np.sum(self.coverage_h_c - self.coverage_l_c)
        #scores = self.confQuantAdapt_c
        #print('A')
        #print(scores)
        #print('B')

        #return XX, scores    


# TEST
#if __name__ == '__main__':
    #for i in range(config.trial):
        #data_path = 'exp_data/' + 'dim=' + str(config.input_dim) + '/' + str(config.noise_type) + '/' + str(config.outlier_type) + '/Iter=' + str(config.Iter) + '/trial=' + str(i+1) + '/' 
        #observation = np.load(data_path + 'outlier.npz')
        #noise = np.load(data_path + 'noise.npz')
        #data = np.load(data_path + 'data.npz')
        
        #if eval('address.' + config.method)['processing'] == 'batch':
            #learn = batch_learning(observation=observation, noise=noise, data=data, alpha=config.alpha, method=eval('address.' + config.method), trial=i+1)
            #grd_truth = gt(data_path=learn.data_path, observation=observation, noise=noise, data=data, alpha=config.alpha)
            #learn.pre_learning()
            #learn.learning()
            #learn.eval(ground_truth=grd_truth)
            #learn.save()
            
        #elif eval('address.' + config.method)['processing'] == 'online':
            #learn = online_learning(observation=observation, noise=noise, data=data, alpha=config.alpha, method=eval('address.' + config.method), trial=i+1)
            #grd_truth = gt(data_path=learn.data_path, observation=observation, noise=noise, data=data, alpha=config.alpha)
            #learn.pre_learning()
            #learn.learning(loss=eval('address.' + config.loss))
            #learn.eval(ground_truth=grd_truth)
            #learn.save()
        #else:
            #print('ERROR: You should choose a method correctly.')
        