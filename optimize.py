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

class base_learning():
    def __init__(self, observation, noise, data, alpha, method, trial, outlier_rate):
    #def __init__(self, observation, noise, data, alpha, method, trial, outlier_rate, observation_c, data_c):
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

        #self.output_c = observation_c['output_test']
        #self.input_c = data_c['input_test']
        
        self.alpha = alpha
        self.Iter = len(self.output_test)
        self.method = method
        
        self.data_path_temp = 'result/text/dim=' + str(len(self.input_train[0])) + '/' + str(noise['noise_type']) + '/' + str(observation['outlier_type']) + '/outlier_rate=' + str(outlier_rate) + '/Iter=' + str(self.Iter) + '/alpha=' + str(alpha_range) 
        self.data_path = self.data_path_temp + '/trial=' + str(trial) + '/'
        mkdir(self.data_path, exist_ok=True)
             
    def eval(self, ground_truth):
        self.coverage_all = coverage(func_est=self.func_est, output_test=self.output_test, alpha=self.alpha, Iter=self.Iter, method=self.method)
        # print(self.coverage_all)
        self.coverage = coverage(func_est=self.func_est, output_test=self.output_test_noise, alpha=self.alpha, Iter=self.Iter, method=self.method)
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
        self.range_func_est_ave, self.coverage_db = error(func_est=self.func_est, gt=ground_truth, Iter=self.Iter, method=self.method)
        print("-----------------------------------")
        print("error")
        for i in range(10):
            print(10 * np.log10(self.coverage_db[2, i*num_div:i*num_div+5]).reshape(1, -1))
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
            gd = grad(alpha=self.alpha, loss=loss, Iter=self.Iter, kernel_vector=self.kernel_vector, kernel_vector_eval=self.kernel_vector_eval, output_train=self.output_train)
            self.learned = gd.learning(step_size=config.step_size)
            self.func_est = self.learned[0]
            self.kernel_weight = self.learned[1]
            self.func_est_final = self.func_est[:, - 1, :]
            self.Iter = gd.Iter

    def save(self):
        data_path = self.data_path + '/online/' + str(self.loss['loss']) + '/\u03b3=' + str(self.loss['gamma']) 
        mkdir(data_path, exist_ok=True) 
        data_path = data_path + '/' + str(self.method['save_name']) + '.npz'
        np.savez_compressed(data_path, coverage=self.coverage, coverage_all=self.coverage_all, range_ave=self.range_func_est_ave, coverage_db=self.coverage_db, func_est=self.func_est_final)

class base_learning_lo():
    def __init__(self, observation, noise, data, alpha, method, trial, outlier_rate):
    #def __init__(self, observation, noise, data, alpha, method, trial, outlier_rate, observation_c, data_c):
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
        self.Iter = len(self.output_test)
        self.method = method
        
        self.data_path_temp = 'result/text/dim=' + str(len(self.input_train[0])) + '/' + str(noise['noise_type']) + '/' + str(observation['outlier_type']) + '/outlier_rate=' + str(outlier_rate) + '/Iter=' + str(self.Iter) + '/alpha=' + str(alpha_range)
        self.data_path = self.data_path_temp + '/lo/trial=' + str(trial) + '/'
        mkdir(self.data_path, exist_ok=True)
             
    def eval(self, ground_truth):
        self.coverage_all = coverage(func_est=self.func_est, output_test=self.output_test, alpha=self.alpha, Iter=self.Iter, method=self.method)
        # print(self.coverage_all)
        self.coverage = coverage(func_est=self.func_est, output_test=self.output_test_noise, alpha=self.alpha, Iter=self.Iter, method=self.method)
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
        self.range_func_est_ave, self.coverage_db = error(func_est=self.func_est, gt=ground_truth, Iter=self.Iter, method=self.method)
        print("-----------------------------------")
        print("error")
        for i in range(10):
            print(10 * np.log10(self.coverage_db[2, i*num_div:i*num_div+5]).reshape(1, -1))
        print(np.argmin(self.coverage_db[2]).reshape(1, -1))
        print(np.min(10 * np.log10(self.coverage_db[2])).reshape(1, -1))

class online_learning_lo(base_learning_lo):
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
            gd = grad(alpha=self.alpha, loss=loss, Iter=self.Iter, kernel_vector=self.kernel_vector, kernel_vector_eval=self.kernel_vector_eval, output_train=self.output_train)
            self.learned = gd.learning(step_size=config.step_size)
            self.func_est = self.learned[0]
            self.kernel_weight = self.learned[1]
            self.func_est_final = self.func_est[:, - 1, :]
            self.Iter = gd.Iter

    def save(self):
        data_path = self.data_path + '/online/' + str(self.loss['loss']) + '/\u03b3=' + str(self.loss['gamma']) 
        mkdir(data_path, exist_ok=True) 
        data_path = data_path + '/' + str(self.method['save_name']) + '.npz'
        np.savez_compressed(data_path, coverage=self.coverage, coverage_all=self.coverage_all, range_ave=self.range_func_est_ave, coverage_db=self.coverage_db, func_est=self.func_est_final)

class base_learning_hi():
    def __init__(self, observation, noise, data, alpha, method, trial, outlier_rate):
    #def __init__(self, observation, noise, data, alpha, method, trial, outlier_rate, observation_c, data_c):
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
        self.Iter = len(self.output_test)
        self.method = method
        
        self.data_path_temp = 'result/text/dim=' + str(len(self.input_train[0])) + '/' + str(noise['noise_type']) + '/' + str(observation['outlier_type']) + '/outlier_rate=' + str(outlier_rate) + '/Iter=' + str(self.Iter) + '/alpha=' + str(alpha_range)
        self.data_path = self.data_path_temp + '/hi/trial=' + str(trial) + '/'
        mkdir(self.data_path, exist_ok=True)
             
    def eval(self, ground_truth):
        self.coverage_all = coverage(func_est=self.func_est, output_test=self.output_test, alpha=self.alpha, Iter=self.Iter, method=self.method)
        # print(self.coverage_all)
        self.coverage = coverage(func_est=self.func_est, output_test=self.output_test_noise, alpha=self.alpha, Iter=self.Iter, method=self.method)
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
        self.range_func_est_ave, self.coverage_db = error(func_est=self.func_est, gt=ground_truth, Iter=self.Iter, method=self.method)
        print("-----------------------------------")
        print("error")
        for i in range(10):
            print(10 * np.log10(self.coverage_db[2, i*num_div:i*num_div+5]).reshape(1, -1))
        print(np.argmin(self.coverage_db[2]).reshape(1, -1))
        print(np.min(10 * np.log10(self.coverage_db[2])).reshape(1, -1))

class online_learning_hi(base_learning_hi):
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
            gd = grad(alpha=self.alpha, loss=loss, Iter=self.Iter, kernel_vector=self.kernel_vector, kernel_vector_eval=self.kernel_vector_eval, output_train=self.output_train)
            self.learned = gd.learning(step_size=config.step_size)
            self.func_est = self.learned[0]
            self.kernel_weight = self.learned[1]
            self.func_est_final = self.func_est[:, - 1, :]
            self.Iter = gd.Iter

    def save(self):
        data_path = self.data_path + '/online/' + str(self.loss['loss']) + '/\u03b3=' + str(self.loss['gamma']) 
        mkdir(data_path, exist_ok=True) 
        data_path = data_path + '/' + str(self.method['save_name']) + '.npz'
        np.savez_compressed(data_path, coverage=self.coverage, coverage_all=self.coverage_all, range_ave=self.range_func_est_ave, coverage_db=self.coverage_db, func_est=self.func_est_final)

class base_learning_hal():
    def __init__(self, observation, noise, data, alpha, method, trial, outlier_rate):
    #def __init__(self, observation, noise, data, alpha, method, trial, outlier_rate, observation_c, data_c):
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
        self.Iter = len(self.output_test)
        self.method = method
        
        self.data_path_temp = 'result/text/dim=' + str(len(self.input_train[0])) + '/' + str(noise['noise_type']) + '/' + str(observation['outlier_type']) + '/outlier_rate=' + str(outlier_rate) + '/Iter=' + str(self.Iter) + '/alpha=' + str(alpha_range)
        self.data_path = self.data_path_temp + '/hal/trial=' + str(trial) + '/'
        mkdir(self.data_path, exist_ok=True)
             
    def eval(self, ground_truth):
        self.coverage_all = coverage(func_est=self.func_est, output_test=self.output_test, alpha=self.alpha, Iter=self.Iter, method=self.method)
        # print(self.coverage_all)
        self.coverage = coverage(func_est=self.func_est, output_test=self.output_test_noise, alpha=self.alpha, Iter=self.Iter, method=self.method)
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
        self.range_func_est_ave, self.coverage_db = error(func_est=self.func_est, gt=ground_truth, Iter=self.Iter, method=self.method)
        print("-----------------------------------")
        print("error")
        for i in range(10):
            print(10 * np.log10(self.coverage_db[2, i*num_div:i*num_div+5]).reshape(1, -1))
        print(np.argmin(self.coverage_db[2]).reshape(1, -1))
        print(np.min(10 * np.log10(self.coverage_db[2])).reshape(1, -1))

class online_learning_hal(base_learning_hal):
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
            gd = grad(alpha=self.alpha, loss=loss, Iter=self.Iter, kernel_vector=self.kernel_vector, kernel_vector_eval=self.kernel_vector_eval, output_train=self.output_train)
            self.learned = gd.learning(step_size=config.step_size)
            self.func_est = self.learned[0]
            self.kernel_weight = self.learned[1]
            self.func_est_final = self.func_est[:, - 1, :]
            self.Iter = gd.Iter

    def save(self):
        data_path = self.data_path + '/online/' + str(self.loss['loss']) + '/\u03b3=' + str(self.loss['gamma']) 
        mkdir(data_path, exist_ok=True) 
        data_path = data_path + '/' + str(self.method['save_name']) + '.npz'
        np.savez_compressed(data_path, coverage=self.coverage, coverage_all=self.coverage_all, range_ave=self.range_func_est_ave, coverage_db=self.coverage_db, func_est=self.func_est_final)

# TEST
if __name__ == '__main__':
    for i in range(config.trial):
        data_path = 'exp_data/' + 'dim=' + str(config.input_dim) + '/' + str(config.noise_type) + '/' + str(config.outlier_type) + '/Iter=' + str(config.Iter) + '/trial=' + str(i+1) + '/' 
        observation = np.load(data_path + 'outlier.npz')
        noise = np.load(data_path + 'noise.npz')
        data = np.load(data_path + 'data.npz')
          
        if eval('address.' + config.method)['processing'] == 'online':
            learn = online_learning(observation=observation, noise=noise, data=data, alpha=config.alpha, method=eval('address.' + config.method), trial=i+1)
            grd_truth = gt(data_path=learn.data_path, observation=observation, noise=noise, data=data, alpha=config.alpha)
            learn.pre_learning()
            learn.learning(loss=eval('address.' + config.loss))
            learn.eval(ground_truth=grd_truth)
            learn.save()
        else:
            print('ERROR: You should choose a method correctly.')
        