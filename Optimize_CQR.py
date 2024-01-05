import numpy as np
import configuration.config as config
import configuration.address as address 
from algorithms import *
from os import makedirs as mkdir
from range_get import range_get

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
    return sr2

def gtCQR(data_path, observation, noise, data, alpha):
    true_a, true_b, true_c = np.array_split(data['output_true_test'], 3, 0)
    obse_a, obse_b, obse_c = np.array_split(observation['output_test'], 3, 0)
    noise_a,noise_b,noise_c = np.array_split(noise['noise_test'], 3, 0)
    #sr, grd_truth, same_range, range_gt_ave = ground_truth(output_true_test=true_c, output_test=obse_c, noise=noise_c, alpha=alpha)
    true_x, true_d = np.array_split(true_c, 2, 0)
    obse_x, obse_d = np.array_split(obse_c, 2, 0)
    noise_x,noise_d = np.array_split(noise_c, 2, 0)
    sr, grd_truth, same_range, range_gt_ave = ground_truth(output_true_test=true_d, output_test=obse_d, noise=noise_d, alpha=alpha)
    sr2, grd_truth2, same_range2, range_gt_ave2 = ground_truth(output_true_test=data['output_true_test'], output_test=observation['output_test'], noise=noise['noise_test'], alpha=alpha)
    data_path_temp = data_path + '/base/CQR'
    mkdir(data_path_temp, exist_ok=True)
    data_path = data_path_temp + '/' + str(address.same_range['save_name']) + '.npz'
    np.savez_compressed(data_path, func_est=sr, range_ave=same_range)
    data_path = data_path_temp + '/' + str(address.ground_truth['save_name']) + '.npz'
    np.savez_compressed(data_path, func_est=grd_truth, range_ave=range_gt_ave)
    data_path = data_path_temp + '/exp_data.npz'
    np.savez_compressed(data_path, input_train=data['input_train'], input_test=data['input_test'], output_true_train=data['output_true_train'], output_true_test=data['output_true_test'], observation_train=observation['output_train'], observation_test=observation['output_test']) 
    return sr, sr2

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
        mkdir(self.data_path, exist_ok=True)
             
    def eval(self, ground_truth):
        self.coverage_all = coverage(func_est=self.func_est_final, output_test=self.output_te, alpha=self.alpha, Iter=self.Iter, method=self.method)
        # print(self.coverage_all)
        #self.coverage = coverage(func_est=self.func_est_final, output_test=self.output_te, alpha=self.alpha, Iter=self.Iter, method=self.method)
        self.coverage = coverage(func_est=self.func_est_ul, output_test=self.output_test, alpha=self.alpha, Iter=self.Iter, method=self.method)
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
        #self.range_func_est_ave, self.coverage_db = error(func_est=self.func_est_final, gt=ground_truth, Iter=self.Iter, method=self.method)
        self.range_func_est_ave, self.coverage_db = error(func_est=self.func_est_ul, gt=ground_truth, Iter=self.Iter, method=self.method)
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
            print('mise')
            print(self.Iter)
            #トレーニングセット，キャリブレーションセット，テストセットに分割
            #self.input_a, self.input_b, self.input_te = np.array_split(self.input_test, 3, 0)
            #self.input_c = np.vstack((self.input_a, self.input_b))
            #self.output_a, self.output_b, self.output_te = np.array_split(self.output_test, 3, 0)
            #self.output_c = np.vstack((self.output_a, self.output_b))
            #trainPoints = np.random.choice(np.arange(self.Iter*2), size=int(self.Iter), replace=False)
            #calpoints = np.delete(np.arange(self.Iter*2), trainPoints)
            self.input_a, self.input_b, self.input_c = np.array_split(self.input_test, 3, 0)
            self.input_d = np.vstack((self.input_a, self.input_b))
            self.input_e, self.input_f = np.array_split(self.input_c, 2, 0)
            self.input_g = np.vstack((self.input_d, self.input_e))
            self.output_a, self.output_b, self.output_c = np.array_split(self.output_test, 3, 0)
            self.output_d = np.vstack((self.output_a, self.output_b))
            self.output_e, self.output_f = np.array_split(self.output_c, 2, 0)
            #self.output_g = np.vstack((self.output_d, self.output_e))
            #trainPoints = np.random.choice(np.arange(2500), size=2000, replace=False)
            #calpoints = np.delete(np.arange(2500), trainPoints)
            #self.input_tr = self.input_g[trainPoints, :]
            #self.output_tr = self.output_g[trainPoints]
            #self.input_ca = self.input_g[calpoints, :]
            #self.output_ca = self.output_g[calpoints]
            self.input_tr = self.input_d
            self.output_tr = self.output_d
            self.input_ca = self.input_e
            self.output_ca = self.output_e
            self.input_te = self.input_f
            self.output_te = self.output_f
            self.Iter_tr = int(len(self.input_tr))
            #data_path = data_path_temp + '/exp_data.npz'
            #トレーニングセットでカーネル計算
            ol_tr = eval(self.method['method'])(input=self.input_tr, dict_band=self.method['dict_band'])
            ol_tr.dict_define(self.method['variable'])
            self.train_vector = ol_tr.kernel_vector(self.input_tr)
            #トレーニングセットで重み計算
            gd = grad(alpha=self.alpha, loss=loss, Iter=self.Iter_tr, kernel_vector=self.train_vector, kernel_vector_eval=self.train_vector, output_train=self.output_tr)
            self.learned = gd.learning(step_size=config.step_size)
            self.func_est = self.learned[0]
            self.func_est_semi = self.func_est[:, - 1, :]
            self.func_est_fin = np.zeros([len(self.alpha), 1, len(self.output_tr)])      
            for a in range(len(self.alpha)):
                self.func_est_fin[a,:,:] = self.func_est_semi[a]
            self.kernel_weight = self.learned[1]

            #おまけ
            #savepath = 'alpha'
            #self.func_est_fin[0] = self.func_est_fin[0].reshape(-1)
            #self.func_est_fin[1] = self.func_est_fin[1].reshape(-1)
            #range_get(input_test=self.input_tr, func_est=self.func_est_fin, savepath=savepath)

            #キャリブレーションセットでカーネル計算
            #ol_c = eval(self.method['method'])(input=self.input_ca, dict_band=self.method['dict_band'])
            self.calib_vector = ol_tr.kernel_vector(self.input_ca)
            #キャリブレーションセットで区間構築
            self.func_calib = np.zeros([len(self.alpha), 1, len(self.output_ca)])      
            for a in range(len(self.alpha)):
                self.func_calib[a,:,:] = np.dot(self.kernel_weight[a].T, self.calib_vector)
            
            #おまけ
            #self.input_ca = self.input_ca.reshape(-1)
            #self.func_calib[0] = self.func_calib[0].reshape(-1)
            #self.func_calib[1] = self.func_calib[1].reshape(-1)
            #savepath = 'beta'
            #range_get(input_test=self.input_ca, func_est=self.func_calib, savepath=savepath)

            #for i in range(self.Iter):        
                # Pinball Moreau
            #キャリブレーションセットで適合性スコア計算
            self.func_low_c = self.func_calib[0].T
            self.func_high_c = self.func_calib[1].T
            self.scores_c = np.maximum(self.output_ca - self.func_high_c.reshape(-1, 1), self.func_low_c.reshape(-1, 1) - self.output_ca)
            self.confQuantAdapt_c = np.percentile(self.scores_c, config.alpha_range * 100)
            #print('pray')
            #print(self.scores_c)
            #print(config.alpha_range)
            #print(self.confQuantAdapt_c)            
            self.X_c = np.full([len(self.scores_c), 1], self.confQuantAdapt_c)

            #テストセットでカーネル計算
            self.test_vector = ol_tr.kernel_vector(self.input_te)
            #テストセットで区間構築
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
            
            #self.func_est = self.learned[0]
            self.Y_c = np.full([len(self.output_tr), 1], self.confQuantAdapt_c)
            self.Z_c = np.full([len(self.output_ca), 1], self.confQuantAdapt_c)
            self.func_low_tr = self.func_est_fin[0].T - self.Y_c.reshape(-1, 1)
            self.func_high_tr = self.func_est_fin[1].T + self.Y_c.reshape(-1, 1)
            self.func_low_c = self.func_calib[0].T - self.Z_c.reshape(-1, 1)
            self.func_high_c = self.func_calib[1].T + self.Z_c.reshape(-1, 1)
            self.func_est_tr = np.hstack((self.func_low_tr, self.func_high_tr)).T
            self.func_est_c = np.hstack((self.func_low_c, self.func_high_c)).T
            self.func_est_ul = np.hstack((self.func_est_tr, self.func_est_c, self.func_est_final))
            #おまけ
            #self.input_te = self.input_te.reshape(-1)
            #self.func_est_final[0] = self.func_est_final[0].reshape(-1)
            #self.func_est_final[1] = self.func_est_final[1].reshape(-1)
            #savepath = 'gamma'
            #range_get(input_test=self.input_te, func_est=self.func_est_final, savepath=savepath)

    def save(self):
        data_path = self.data_path + '/online/' + str(self.loss['loss']) + '/\u03b3=' + str(self.loss['gamma']) + '/CQR' 
        mkdir(data_path, exist_ok=True) 
        data_path = data_path + '/' + str(self.method['save_name']) + '.npz'
        print(data_path)
        np.savez_compressed(data_path, coverage=self.coverage, coverage_all=self.coverage_all, range_ave=self.range_func_est_ave, coverage_db=self.coverage_db, func_est=self.func_est_final, func_est_all=self.func_est_ul, input_te = self.input_te, input_test = self.input_test)
   
class base_learning_lo():
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
        self.data_path = self.data_path_temp + '/lo/trial=' + str(trial) + '/'
        mkdir(self.data_path, exist_ok=True)
             
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
        self.range_func_est_ave, self.coverage_db = error(func_est=self.func_est_ul, gt=ground_truth, Iter=self.Iter, method=self.method)
        print("-----------------------------------")
        print("error")
        print(10 * np.log10(self.coverage_db[2]).reshape(1, -1))
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
            print('mise')
            print(self.Iter)
            #トレーニングセット，キャリブレーションセット，テストセットに分割
            #self.input_a, self.input_b, self.input_te = np.array_split(self.input_test, 3, 0)
            #self.input_c = np.vstack((self.input_a, self.input_b))
            #self.output_a, self.output_b, self.output_te = np.array_split(self.output_test, 3, 0)
            #self.output_c = np.vstack((self.output_a, self.output_b))
            #trainPoints = np.random.choice(np.arange(self.Iter*2), size=int(self.Iter), replace=False)
            #calpoints = np.delete(np.arange(self.Iter*2), trainPoints)
            self.input_a, self.input_b, self.input_c = np.array_split(self.input_test, 3, 0)
            self.input_d = np.vstack((self.input_a, self.input_b))
            self.input_e, self.input_f = np.array_split(self.input_c, 2, 0)
            self.input_g = np.vstack((self.input_d, self.input_e))
            self.output_a, self.output_b, self.output_c = np.array_split(self.output_test, 3, 0)
            self.output_d = np.vstack((self.output_a, self.output_b))
            self.output_e, self.output_f = np.array_split(self.output_c, 2, 0)
            #self.output_g = np.vstack((self.output_d, self.output_e))
            #trainPoints = np.random.choice(np.arange(2500), size=2000, replace=False)
            #calpoints = np.delete(np.arange(2500), trainPoints)
            #self.input_tr = self.input_g[trainPoints, :]
            #self.output_tr = self.output_g[trainPoints]
            #self.input_ca = self.input_g[calpoints, :]
            #self.output_ca = self.output_g[calpoints]
            self.input_tr = self.input_d
            self.output_tr = self.output_d
            self.input_ca = self.input_e
            self.output_ca = self.output_e
            self.input_te = self.input_f
            self.output_te = self.output_f
            self.Iter_tr = int(len(self.input_tr))
            #data_path = data_path_temp + '/exp_data.npz'
            #トレーニングセットでカーネル計算
            ol_tr = eval(self.method['method'])(input=self.input_tr, dict_band=self.method['dict_band'])        
            ol_tr.dict_define(self.method['variable'])
            self.train_vector = ol_tr.kernel_vector(self.input_tr)
            #トレーニングセットで重み計算
            gd = grad(alpha=self.alpha, loss=loss, Iter=self.Iter_tr, kernel_vector=self.train_vector, kernel_vector_eval=self.train_vector, output_train=self.output_tr)
            self.learned = gd.learning(step_size=config.step_size)
            self.func_est = self.learned[0]
            self.func_est_semi = self.func_est[:, - 1, :]
            self.func_est_fin = np.zeros([len(self.alpha), 1, len(self.output_tr)])      
            for a in range(len(self.alpha)):
                self.func_est_fin[a,:,:] = self.func_est_semi[a]
            #self.func_est_fin = self.func_est[:, - 1, :]
            self.kernel_weight = self.learned[1]

            #おまけ
            #savepath = 'alpha'
            #self.func_est_fin[0] = self.func_est_fin[0].reshape(-1)
            #self.func_est_fin[1] = self.func_est_fin[1].reshape(-1)
            #range_get(input_test=self.input_tr, func_est=self.func_est_fin, savepath=savepath)

            #キャリブレーションセットでカーネル計算
            #ol_c = eval(self.method['method'])(input=self.input_ca, dict_band=self.method['dict_band'])
            self.calib_vector = ol_tr.kernel_vector(self.input_ca)
            #キャリブレーションセットで区間構築
            self.func_calib = np.zeros([len(self.alpha), 1, len(self.output_ca)])      
            for a in range(len(self.alpha)):
                self.func_calib[a,:,:] = np.dot(self.kernel_weight[a].T, self.calib_vector)
            
            #おまけ
            #self.input_ca = self.input_ca.reshape(-1)
            #self.func_calib[0] = self.func_calib[0].reshape(-1)
            #self.func_calib[1] = self.func_calib[1].reshape(-1)
            #savepath = 'beta'
            #range_get(input_test=self.input_ca, func_est=self.func_calib, savepath=savepath)

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
            #テストセットで区間構築
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

            self.Y_c = np.full([len(self.output_tr), 1], self.confQuantAdapt_c)
            self.Z_c = np.full([len(self.output_ca), 1], self.confQuantAdapt_c)
            self.func_low_tr = self.func_est_fin[0].T - self.Y_c.reshape(-1, 1)
            self.func_high_tr = self.func_est_fin[1].T + self.Y_c.reshape(-1, 1)
            self.func_low_c = self.func_calib[0].T - self.Z_c.reshape(-1, 1)
            self.func_high_c = self.func_calib[1].T + self.Z_c.reshape(-1, 1)
            self.func_est_tr = np.hstack((self.func_low_tr, self.func_high_tr)).T
            self.func_est_c = np.hstack((self.func_low_c, self.func_high_c)).T
            self.func_est_ul = np.hstack((self.func_est_tr, self.func_est_c, self.func_est_final))
            #おまけ
            #self.input_te = self.input_te.reshape(-1)
            #self.func_est_final[0] = self.func_est_final[0].reshape(-1)
            #self.func_est_final[1] = self.func_est_final[1].reshape(-1)
            #savepath = 'gamma'
            #range_get(input_test=self.input_te, func_est=self.func_est_final, savepath=savepath)

    def save(self):
        data_path = self.data_path + '/online/' + str(self.loss['loss']) + '/\u03b3=' + str(self.loss['gamma']) + '/CQR' 
        mkdir(data_path, exist_ok=True) 
        data_path = data_path + '/' + str(self.method['save_name']) + '.npz'
        print(data_path)
        np.savez_compressed(data_path, coverage=self.coverage, coverage_all=self.coverage_all, range_ave=self.range_func_est_ave, coverage_db=self.coverage_db, func_est=self.func_est_final, func_est_all=self.func_est_ul,  input_te = self.input_te, input_test = self.input_test)

class base_learning_hi():
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
        self.data_path = self.data_path_temp + '/hi/trial=' + str(trial) + '/'
        mkdir(self.data_path, exist_ok=True)
             
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
        self.range_func_est_ave, self.coverage_db = error(func_est=self.func_est_ul, gt=ground_truth, Iter=self.Iter, method=self.method)
        print("-----------------------------------")
        print("error")
        print(10 * np.log10(self.coverage_db[2]).reshape(1, -1))
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
            print('mise')
            print(self.Iter)
            #トレーニングセット，キャリブレーションセット，テストセットに分割
            #self.input_a, self.input_b, self.input_te = np.array_split(self.input_test, 3, 0)
            #self.input_c = np.vstack((self.input_a, self.input_b))
            #self.output_a, self.output_b, self.output_te = np.array_split(self.output_test, 3, 0)
            #self.output_c = np.vstack((self.output_a, self.output_b))
            #trainPoints = np.random.choice(np.arange(self.Iter*2), size=int(self.Iter), replace=False)
            #calpoints = np.delete(np.arange(self.Iter*2), trainPoints)
            self.input_a, self.input_b, self.input_c = np.array_split(self.input_test, 3, 0)
            self.input_d = np.vstack((self.input_a, self.input_b))
            self.input_e, self.input_f = np.array_split(self.input_c, 2, 0)
            self.input_g = np.vstack((self.input_d, self.input_e))
            self.output_a, self.output_b, self.output_c = np.array_split(self.output_test, 3, 0)
            self.output_d = np.vstack((self.output_a, self.output_b))
            self.output_e, self.output_f = np.array_split(self.output_c, 2, 0)
            #self.output_g = np.vstack((self.output_d, self.output_e))
            #trainPoints = np.random.choice(np.arange(2500), size=2000, replace=False)
            #calpoints = np.delete(np.arange(2500), trainPoints)
            #self.input_tr = self.input_g[trainPoints, :]
            #self.output_tr = self.output_g[trainPoints]
            #self.input_ca = self.input_g[calpoints, :]
            #self.output_ca = self.output_g[calpoints]
            self.input_tr = self.input_d
            self.output_tr = self.output_d
            self.input_ca = self.input_e
            self.output_ca = self.output_e
            self.input_te = self.input_f
            self.output_te = self.output_f
            self.Iter_tr = int(len(self.input_tr))
            #data_path = data_path_temp + '/exp_data.npz'
            #トレーニングセットでカーネル計算
            ol_tr = eval(self.method['method'])(input=self.input_tr, dict_band=self.method['dict_band'])        
            ol_tr.dict_define(self.method['variable'])
            self.train_vector = ol_tr.kernel_vector(self.input_tr)
            #トレーニングセットで重み計算
            gd = grad(alpha=self.alpha, loss=loss, Iter=self.Iter_tr, kernel_vector=self.train_vector, kernel_vector_eval=self.train_vector, output_train=self.output_tr)
            self.learned = gd.learning(step_size=config.step_size)
            self.func_est = self.learned[0]
            self.func_est_semi = self.func_est[:, - 1, :]
            self.func_est_fin = np.zeros([len(self.alpha), 1, len(self.output_tr)])      
            for a in range(len(self.alpha)):
                self.func_est_fin[a,:,:] = self.func_est_semi[a]
            #self.func_est_fin = self.func_est[:, - 1, :]
            self.kernel_weight = self.learned[1]

            #おまけ
            #savepath = 'alpha'
            #self.func_est_fin[0] = self.func_est_fin[0].reshape(-1)
            #self.func_est_fin[1] = self.func_est_fin[1].reshape(-1)
            #range_get(input_test=self.input_tr, func_est=self.func_est_fin, savepath=savepath)

            #キャリブレーションセットでカーネル計算
            #ol_c = eval(self.method['method'])(input=self.input_ca, dict_band=self.method['dict_band'])
            self.calib_vector = ol_tr.kernel_vector(self.input_ca)
            #キャリブレーションセットで区間構築
            self.func_calib = np.zeros([len(self.alpha), 1, len(self.output_ca)])      
            for a in range(len(self.alpha)):
                self.func_calib[a,:,:] = np.dot(self.kernel_weight[a].T, self.calib_vector)
            
            #おまけ
            #self.input_ca = self.input_ca.reshape(-1)
            #self.func_calib[0] = self.func_calib[0].reshape(-1)
            #self.func_calib[1] = self.func_calib[1].reshape(-1)
            #savepath = 'beta'
            #range_get(input_test=self.input_ca, func_est=self.func_calib, savepath=savepath)

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
            #テストセットで区間構築
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

            self.Y_c = np.full([len(self.output_tr), 1], self.confQuantAdapt_c)
            self.Z_c = np.full([len(self.output_ca), 1], self.confQuantAdapt_c)
            self.func_low_tr = self.func_est_fin[0].T - self.Y_c.reshape(-1, 1)
            self.func_high_tr = self.func_est_fin[1].T + self.Y_c.reshape(-1, 1)
            self.func_low_c = self.func_calib[0].T - self.Z_c.reshape(-1, 1)
            self.func_high_c = self.func_calib[1].T + self.Z_c.reshape(-1, 1)
            self.func_est_tr = np.hstack((self.func_low_tr, self.func_high_tr)).T
            self.func_est_c = np.hstack((self.func_low_c, self.func_high_c)).T
            self.func_est_ul = np.hstack((self.func_est_tr, self.func_est_c, self.func_est_final))
            #おまけ
            #self.input_te = self.input_te.reshape(-1)
            #self.func_est_final[0] = self.func_est_final[0].reshape(-1)
            #self.func_est_final[1] = self.func_est_final[1].reshape(-1)
            #savepath = 'gamma'
            #range_get(input_test=self.input_te, func_est=self.func_est_final, savepath=savepath)

    def save(self):
        data_path = self.data_path + '/online/' + str(self.loss['loss']) + '/\u03b3=' + str(self.loss['gamma']) + '/CQR' 
        mkdir(data_path, exist_ok=True) 
        data_path = data_path + '/' + str(self.method['save_name']) + '.npz'
        print(data_path)
        np.savez_compressed(data_path, coverage=self.coverage, coverage_all=self.coverage_all, range_ave=self.range_func_est_ave, coverage_db=self.coverage_db, func_est=self.func_est_final, func_est_all=self.func_est_ul, input_te = self.input_te, input_test = self.input_test)


class base_learning_hal():
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
        self.data_path = self.data_path_temp + '/hal/trial=' + str(trial) + '/'
        mkdir(self.data_path, exist_ok=True)
             
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
        self.range_func_est_ave, self.coverage_db = error(func_est=self.func_est_ul, gt=ground_truth, Iter=self.Iter, method=self.method)
        print("-----------------------------------")
        print("error")
        print(10 * np.log10(self.coverage_db[2]).reshape(1, -1))
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
            print('mise')
            print(self.Iter)
            #トレーニングセット，キャリブレーションセット，テストセットに分割
            #self.input_a, self.input_b, self.input_te = np.array_split(self.input_test, 3, 0)
            #self.input_c = np.vstack((self.input_a, self.input_b))
            #self.output_a, self.output_b, self.output_te = np.array_split(self.output_test, 3, 0)
            #self.output_c = np.vstack((self.output_a, self.output_b))
            #trainPoints = np.random.choice(np.arange(self.Iter*2), size=int(self.Iter), replace=False)
            #calpoints = np.delete(np.arange(self.Iter*2), trainPoints)
            self.input_a, self.input_b, self.input_c = np.array_split(self.input_test, 3, 0)
            self.input_d = np.vstack((self.input_a, self.input_b))
            self.input_e, self.input_f = np.array_split(self.input_c, 2, 0)
            self.input_g = np.vstack((self.input_d, self.input_e))
            self.output_a, self.output_b, self.output_c = np.array_split(self.output_test, 3, 0)
            self.output_d = np.vstack((self.output_a, self.output_b))
            self.output_e, self.output_f = np.array_split(self.output_c, 2, 0)
            #self.output_g = np.vstack((self.output_d, self.output_e))
            #trainPoints = np.random.choice(np.arange(2500), size=2000, replace=False)
            #calpoints = np.delete(np.arange(2500), trainPoints)
            #self.input_tr = self.input_g[trainPoints, :]
            #self.output_tr = self.output_g[trainPoints]
            #self.input_ca = self.input_g[calpoints, :]
            #self.output_ca = self.output_g[calpoints]
            self.input_tr = self.input_d
            self.output_tr = self.output_d
            self.input_ca = self.input_e
            self.output_ca = self.output_e
            self.input_te = self.input_f
            self.output_te = self.output_f
            self.Iter_tr = int(len(self.input_tr))
            #data_path = data_path_temp + '/exp_data.npz'
            #トレーニングセットでカーネル計算
            ol_tr = eval(self.method['method'])(input=self.input_tr, dict_band=self.method['dict_band'])        
            ol_tr.dict_define(self.method['variable'])
            self.train_vector = ol_tr.kernel_vector(self.input_tr)
            #トレーニングセットで重み計算
            gd = grad(alpha=self.alpha, loss=loss, Iter=self.Iter_tr, kernel_vector=self.train_vector, kernel_vector_eval=self.train_vector, output_train=self.output_tr)
            self.learned = gd.learning(step_size=config.step_size)
            self.func_est = self.learned[0]
            self.func_est_semi = self.func_est[:, - 1, :]
            self.func_est_fin = np.zeros([len(self.alpha), 1, len(self.output_tr)])      
            for a in range(len(self.alpha)):
                self.func_est_fin[a,:,:] = self.func_est_semi[a]
            #self.func_est_fin = self.func_est[:, - 1, :]
            self.kernel_weight = self.learned[1]

            #おまけ
            #savepath = 'alpha'
            #self.func_est_fin[0] = self.func_est_fin[0].reshape(-1)
            #self.func_est_fin[1] = self.func_est_fin[1].reshape(-1)
            #range_get(input_test=self.input_tr, func_est=self.func_est_fin, savepath=savepath)

            #キャリブレーションセットでカーネル計算
            #ol_c = eval(self.method['method'])(input=self.input_ca, dict_band=self.method['dict_band'])
            self.calib_vector = ol_tr.kernel_vector(self.input_ca)
            #キャリブレーションセットで区間構築
            self.func_calib = np.zeros([len(self.alpha), 1, len(self.output_ca)])      
            for a in range(len(self.alpha)):
                self.func_calib[a,:,:] = np.dot(self.kernel_weight[a].T, self.calib_vector)
            
            #おまけ
            #self.input_ca = self.input_ca.reshape(-1)
            #self.func_calib[0] = self.func_calib[0].reshape(-1)
            #self.func_calib[1] = self.func_calib[1].reshape(-1)
            #savepath = 'beta'
            #range_get(input_test=self.input_ca, func_est=self.func_calib, savepath=savepath)

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
            #テストセットで区間構築
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

            self.Y_c = np.full([len(self.output_tr), 1], self.confQuantAdapt_c)
            self.Z_c = np.full([len(self.output_ca), 1], self.confQuantAdapt_c)
            self.func_low_tr = self.func_est_fin[0].T - self.Y_c.reshape(-1, 1)
            self.func_high_tr = self.func_est_fin[1].T + self.Y_c.reshape(-1, 1)
            self.func_low_c = self.func_calib[0].T - self.Z_c.reshape(-1, 1)
            self.func_high_c = self.func_calib[1].T + self.Z_c.reshape(-1, 1)
            self.func_est_tr = np.hstack((self.func_low_tr, self.func_high_tr)).T
            self.func_est_c = np.hstack((self.func_low_c, self.func_high_c)).T
            self.func_est_ul = np.hstack((self.func_est_tr, self.func_est_c, self.func_est_final))
            #おまけ
            #self.input_te = self.input_te.reshape(-1)
            #self.func_est_final[0] = self.func_est_final[0].reshape(-1)
            #self.func_est_final[1] = self.func_est_final[1].reshape(-1)
            #savepath = 'gamma'
            #range_get(input_test=self.input_te, func_est=self.func_est_final, savepath=savepath)

    def save(self):
        data_path = self.data_path + '/online/' + str(self.loss['loss']) + '/\u03b3=' + str(self.loss['gamma']) + '/CQR' 
        mkdir(data_path, exist_ok=True) 
        data_path = data_path + '/' + str(self.method['save_name']) + '.npz'
        print(data_path)
        np.savez_compressed(data_path, coverage=self.coverage, coverage_all=self.coverage_all, range_ave=self.range_func_est_ave, coverage_db=self.coverage_db, func_est=self.func_est_final, func_est_all=self.func_est_ul, input_te = self.input_te, input_test = self.input_test)