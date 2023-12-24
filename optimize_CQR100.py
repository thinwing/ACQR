import numpy as np
import configuration.config as config
import configuration.address as address 
import random
from algorithms import *
from os import makedirs as mkdir

def gt(data_path, observation, noise, data, alpha):
    sr, grd_truth, same_range, range_gt_ave = ground_truth(output_true_test=data['output_true_test'], output_test=observation['output_test'], noise=noise['noise_test'], alpha=alpha)    
    data_path_temp = data_path + '/base'
    mkdir(data_path_temp, exist_ok=True)
    data_path = data_path_temp + '/' + str(address.same_range['save_name']) + '.npz'
    np.savez_compressed(data_path, func_est=sr, range_ave=same_range)
    data_path = data_path_temp + '/' + str(address.ground_truth['save_name']) + '.npz'
    np.savez_compressed(data_path, func_est=grd_truth, range_ave=range_gt_ave)
    data_path = data_path_temp + '/exp_data.npz'
    np.savez_compressed(data_path, input_train=data['input_train'], input_test=data['input_test'], output_true_train=data['output_true_train'], output_true_test=data['output_true_test'], observation_train=observation['output_train'], observation_test=observation['output_test']) 
   
    return grd_truth

def gtCQR(data_path, observation, noise, data, alpha):
    true_a, true_b, true_c = np.array_split(data['output_true_test'], 3, 0)
    obse_a, obse_b, obse_c = np.array_split(observation['output_test'], 3, 0)
    noise_a,noise_b,noise_c = np.array_split(noise['noise_test'], 3, 0)
    sr, grd_truth, same_range, range_gt_ave = ground_truth(output_true_test=true_c, output_test=obse_c, noise=noise_c, alpha=alpha)
    data_path_temp = data_path + '/base/CQR100'
    mkdir(data_path_temp, exist_ok=True)
    data_path = data_path_temp + '/' + str(address.same_range['save_name']) + '.npz'
    np.savez_compressed(data_path, func_est=sr, range_ave=same_range)
    data_path = data_path_temp + '/' + str(address.ground_truth['save_name']) + '.npz'
    np.savez_compressed(data_path, func_est=grd_truth, range_ave=range_gt_ave)
    data_path = data_path_temp + '/exp_data.npz'
    np.savez_compressed(data_path, input_train=data['input_train'], input_test=data['input_test'], output_true_train=data['output_true_train'], output_true_test=data['output_true_test'], observation_train=observation['output_train'], observation_test=observation['output_test']) 
    return grd_truth

   
class base_learning():
    def __init__(self, observation, noise, data, alpha, method, trial, outlier_rate):
        # Data LOADING
        self.output_train = observation['output_train']
        self.output_test = observation['output_test']
        alpha_range = round(alpha[1] - alpha[0], 3)
        self.input_train = data['input_train']
        self.output_true_train = data['output_true_train']
        
        self.input_test = data['input_test']
        self.output_true_test = data['output_true_test']
        
        self.output_test_noise = observation['output_test_noise']

        self.noise_max = noise['noise_test']
        
        self.alpha = alpha
        self.Iter = int(len(self.output_test)/3)
        self.method = method
        
        self.data_path_temp = 'result/text/dim=' + str(len(self.input_train[0])) + '/' + str(noise['noise_type']) + '/' + str(observation['outlier_type']) + '/outlier_rate=' + str(outlier_rate) + '/Iter=' + str(self.Iter) + '/alpha=' + str(alpha_range) 
        self.data_path = self.data_path_temp + '/trial=' + str(trial) + '/'
             
    def eval(self, ground_truth):
        self.coverage_all = coverage(func_est=self.func_est_final, output_test=self.output_te, alpha=self.alpha, Iter=self.Iter, method=self.method)
        # print(self.coverage_all)
        self.coverage = coverage(func_est=self.func_est_final, output_test=self.output_te, alpha=self.alpha, Iter=self.Iter, method=self.method)
        print("-----------------------------------")
        num_div = int(len(self.coverage[0]) / 10)
        print("coverage_")
        for i in range(10):
            print(self.coverage[0,i*num_div:i*num_div+5].reshape(1, -1))
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        for i in range(10):
            print(self.coverage[1,i*num_div:i*num_div+5].reshape(1, -1))

        print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
        print("coverage")
        for i in range(10):
            print((self.coverage[1,i*num_div:i*num_div+5] - self.coverage[0,i*num_div:i*num_div+5]).reshape(1, -1))
        self.range_func_est_ave, self.coverage_db = error(func_est=self.func_est_final, gt=ground_truth, Iter=self.Iter, method=self.method)
        print("-----------------------------------")
        print("error")
        print(10 * np.log10(self.coverage_db[2]).reshape(1, -1))
        print(np.argmin(self.coverage_db[2]).reshape(1, -1))
        print(np.min(10 * np.log10(self.coverage_db[2])).reshape(1, -1))
       
class online_learning(base_learning):
    def pre_learning(self):
        ol = eval(self.method['method'])(input=self.input_train, dict_band=self.method['dict_band'])
        ol.dict_define(self.method['variable'])
        self.kernel_vector = ol.kernel_vector(self.input_train)
        self.kernel_vector_eval = ol.kernel_vector(self.input_test)
        
    def learning(self, loss):
        self.loss = loss

        if loss['loss'] == 'pmc_online':
            prim = primal(alpha=self.alpha, loss=loss , Iter=self.Iter, kernel_vector=self.kernel_vector, kernel_vector_eval=self.kernel_vector_eval, output_train=self.output_train)
            self.func_est = prim.learning(step_size=config.parameter_pdm, regular=config.regular_pdm)
            self.func_est_final = self.func_est[:, - 1, :]
            self.Iter = prim.Iter

        elif loss['loss'] == 'pmc_batch':
            prim = primal_batch(alpha=self.alpha, loss=loss , Iter=config.Iter_batch, kernel_vector=self.kernel_vector, kernel_vector_eval=self.kernel_vector_eval, output_train=self.output_train)
            self.func_est = prim.learning(step_size=config.parameter_pdm, regular=config.regular_pdm)
            self.func_est_final = self.func_est[:, - 1, :]
            self.Iter = prim.Iter

        else:
            #トレーニングセット，キャリブレーションセット，テストセットに分割
            print('Iter')
            print(self.Iter)
            self.number = list(range(config.Iter))
            self.number_shuffle = random.sample(self.number, config.Iter)
            self.number_tr, self.number_ca, self.number_te = np.array_split(self.number_shuffle, 3, 0)
            self.input_tr = self.input_train[self.number_tr, :]
            self.output_tr = self.output_true_train[self.number_tr]
            self.input_ca = self.input_train[self.number_ca, :]
            self.output_ca = self.output_true_train[self.number_ca]
            self.input_te = self.input_train[self.number_te, :]
            self.output_te = self.output_true_train[self.number_te]
            self.Iter_tr = int(len(self.input_tr))
            #トレーニングセットでカーネル計算
            ol_tr = eval(self.method['method'])(input=self.input_tr, dict_band=self.method['dict_band'])        
            ol_tr.dict_define(self.method['variable'])
            self.train_vector = ol_tr.kernel_vector(self.input_tr)
            #トレーニングセットで重み計算
            gd = grad(alpha=self.alpha, loss=loss, Iter=self.Iter_tr, kernel_vector=self.train_vector, kernel_vector_eval=self.train_vector, output_train=self.output_tr)
            self.learned = gd.learning(step_size=config.step_size)
            self.func_est = self.learned[0]
            self.kernel_weight = self.learned[1]
            self.Iter = gd.Iter
            #print('self.Iter')
            #print(self.Iter)
            #キャリブレーションセットでカーネル計算
            self.calib_vector = ol_tr.kernel_vector(self.input_ca)
            #キャリブレーションセットで区間構築
            self.func_calib = np.zeros([len(self.alpha), 1, len(self.output_ca)])
            for a in range(len(self.alpha)):
                self.func_calib[a,:,:] = np.dot(self.kernel_weight[a].T, self.calib_vector)
            #for i in range(self.Iter):        
                # Pinball Moreau
            #キャリブレーションセットで適合性スコア計算
            self.func_low_c = self.func_calib[0].T
            self.func_high_c = self.func_calib[1].T
            self.scores_c = np.maximum(self.output_ca - self.func_high_c.reshape(-1, 1), self.func_low_c.reshape(-1, 1) - self.output_ca)
            self.confQuantAdapt_c = np.percentile(self.scores_c, config.alpha_range * 100)
            self.X_c = np.full([len(self.scores_c), 1], self.confQuantAdapt_c)
            #テストセットでカーネル計算
            self.test_vector = ol_tr.kernel_vector(self.input_te)
            #testセットで区間構築
            self.func_test = np.zeros([len(self.alpha), 1, len(self.output_te)])
            for a in range(len(self.alpha)):
                self.func_test[a,:,:] = np.dot(self.kernel_weight[a].T, self.test_vector)
            self.func_low_t = self.func_test[0].T
            self.func_high_t = self.func_test[1].T            
            self.lower_t = self.func_low_t.reshape(-1, 1) - self.X_c.reshape(-1, 1)
            self.higher_t = self.func_high_t.reshape(-1, 1) + self.X_c.reshape(-1, 1)
            #self.coverage_lt = np.where((self.lower_t - self.output_te > 0), 1, 0)
            #self.coverage_ht = np.where((self.higher_t - self.output_te > 0), 1, 0)
            #XX = np.sum(self.coverage_ht - self.coverage_lt)
            self.func_est_final = np.hstack((self.lower_t, self.higher_t)).T

    def save(self):
        data_path = self.data_path + '/online/' + str(self.loss['loss']) + '/\u03b3=' + str(self.loss['gamma']) + '/CQR100' 
        mkdir(data_path, exist_ok=True)
        data_path = data_path + '/' + str(self.method['save_name']) + '.npz'
        print(data_path)
        np.savez_compressed(data_path, coverage=self.coverage, coverage_all=self.coverage_all, range_ave=self.range_func_est_ave, coverage_db=self.coverage_db, func_est=self.func_est_final,  input_te = self.input_te)