import graph
import data as dt
import optimize
import Optimize_CQR
import Optimize_CQR2
from integrate import data_integrate as integrate
from integrate import data_integrate_CQR as integrate_CQR
from algorithms.online.gradient_descent import online_learning as grad

import datetime
import configuration.config as config
import configuration.graph_config as graph_config
import configuration.address as address


import numpy as np

# data 

alpha_all = config.alpha_all[config.start:config.limit]

# num = config.num_divide
# if num > 0:
#     num_len = int(len(alpha_all_temp) / 3)
#     if num < 3:
#         alpha_all = alpha_all_temp[(num - 1) * num_len : num * num_len]
#     else:
#         alpha_all = alpha_all_temp[(num - 1) * num_len :]
# else:
#     alpha_all = alpha_all_temp
with open('log5.txt', 'a') as f:
    f.write('start')
    f.write('\n---------------------------------------------')

if config.optimize_flag == 'all':
    for noise_type in config.noise_type_all:
        for outlier_type in config.outlier_type_all:
            for outlier_rate in config.outlier_rate:
                for method in config.method_all:
                    for alpha in config.alpha_all:
                        for i in range(config.trial):
                            data_path = 'exp_data/' + 'dim=' + str(config.input_dim) + '/' + str(noise_type) + '/' + str(outlier_type) + '/outlier_rate=' + str(outlier_rate)  +'/Iter=' + str(config.Iter) + '/trial=' + str(i+1) + '/' 
                            observation = np.load(data_path + 'outlier.npz')
                            noise = np.load(data_path + 'noise.npz')
                            data = np.load(data_path + 'data.npz')
                            
                            if eval('address.' + config.method)['processing'] == 'batch':
                                learn = optimize.batch_learning(observation=observation, noise=noise, data=data, alpha=alpha, method=eval('address.' + method), trial=i+1)
                                grd_truth = optimize.gt(data_path=learn.data_path, observation=observation, noise=noise, data=data, alpha=alpha)
                                learn.pre_learning()
                                learn.learning()
                                learn.eval(ground_truth=grd_truth)
                                learn.save()
                                
                            elif eval('address.' + config.method)['processing'] == 'online':
                                learn = optimize.online_learning(observation=observation, noise=noise, data=data, alpha=alpha, method=eval('address.' + method), trial=i+1)
                                grd_truth = optimize.gt(data_path=learn.data_path, observation=observation, noise=noise, data=data, alpha=alpha)
                                learn.pre_learning()
                                for loss in config.loss_all:
                                    learn.learning(loss=eval('address.' + loss))
                                    learn.eval(ground_truth=grd_truth)
                                    learn.save()
                            else:
                                print('ERROR: You should choose a method correctly.')
            
elif config.optimize_flag == 'custom':    
    for noise_type in config.noise_types:
        for outlier_type in config.outlier_types:
            #for outlier_rate in config.outlier_rate:
                outlier_rate = 0.05
                with open('log5.txt', 'a') as f:
                    f.write('\nlo')
                    f.write('\nnoise_type : ' + str(noise_type))
                    f.write('\noutlier_type : ' + str(outlier_type))
                    f.write('\noutlier_rate : ' + str(outlier_rate))
                    f.write('\n---------------------------------------------')

                for index_method, method in enumerate(config.methods):
                    with open('log5.txt', 'a') as f:
                        f.write('\nMethod : ' + str(method))

                    for index_alpha, alpha in enumerate(alpha_all):
                        with open('log5.txt', 'a') as f:
                            f.write('\n---------------------------------------------')
                            f.write('\n' +  str(index_alpha + 1) + ' / ' + str(len(alpha_all)) + ' : ' + str(alpha))
                            f.write('\n---------------------------------------------')
                            
                        for i in range(config.trial):
                            data_path = 'exp_data/' + 'dim=' + str(config.input_dim) + '/' + str(noise_type) + '/' + str(outlier_type) + '/outlier_rate=' + str(outlier_rate) + '/Iter=' + str(config.Iter) + '/trial=' + str(i+1) + '/lo/' 
                            observation = np.load(data_path + 'outlier.npz')
                            noise = np.load(data_path + 'noise.npz')
                            data = np.load(data_path + 'data.npz')
                            #data_c = np.load(data_path + 'calib/data_c.npz')
                            #observation_c = np.load(data_path + 'calib/outlier_c.npz')
                            
                            if eval('address.' + str(method))['processing'] == 'batch':
                                learn = optimize.batch_learning(observation=observation, noise=noise, data=data, alpha=alpha, method=eval('address.' + str(method)), trial=i+1, outlier_rate=outlier_rate)
                                grd_truth = optimize.gt(data_path=learn.data_path, observation=observation, noise=noise, data=data, alpha=alpha)
                                learn.pre_learning()
                                learn.learning()
                                learn.eval(ground_truth=grd_truth)
                                learn.save()
                                
                            elif eval('address.' + str(method))['processing'] == 'online':
                                learn = Optimize_CQR.online_learning(observation=observation, noise=noise, data=data, alpha=alpha, method=eval('address.' + str(method)), trial=i+1, outlier_rate=outlier_rate)
                                grd_truth = Optimize_CQR.gtCQR(data_path=learn.data_path, observation=observation, noise=noise, data=data, alpha=alpha)
                                
                                #truth_path = 'exp_data/dim=1/linear_expansion/sparse/outlier_rate=0.05/Iter=3000/trial=' + str(i+1) + '/outlier.npz'
                                #truth = np.load(truth_path)
                                #grd_truth = Optimize_CQR.gtCQR(data_path=learn.data_path, observation=observation, noise=noise, data=data, alpha=alpha)

                                #learn = Optimize_CQR2.online_learning(observation=observation, noise=noise, data=data, alpha=alpha, method=eval('address.' + str(method)), trial=i+1, outlier_rate=outlier_rate)
                                #grd_truth = Optimize_CQR2.gtCQR(data_path=learn.data_path, observation=observation, noise=noise, data=data, alpha=alpha)
                                learn.pre_learning()
                                for loss_temp in config.losses:
                                    if str(loss_temp) == 'pinball':
                                        loss = eval('address.' + str(loss_temp))
                                        loss['gamma'] = 0
                                        learn.learning(loss=loss)
                                        learn.eval(ground_truth=grd_truth)
                                        #covlen = learn.CQR(data_c=data_c, observation_c=observation_c)
                                        learn.save()
                                        
                                    else:
                                        for gamma in config.gamma:
                                            loss = eval('address.' + str(loss_temp))
                                            loss['gamma'] = gamma
                                            learn.learning(loss=loss)
                                            learn.eval(ground_truth=grd_truth)
                                            learn.save()
                                            now = datetime.datetime.now()
                                            with open('log5.txt', 'a') as f:
                                                f.write('\n' +'gamma = ' + str(gamma))
                                                f.write('\n' + '\t' + str(i + 1) + ' / ' + str(config.trial) + ' : ' + str(now))
                                                f.write('\n' + '\t\tCoverage rate = ' + str((learn.coverage[1] - learn.coverage[0])[-1]))

                            else:
                                print('ERROR: You should choose a method correctly.')

                            now = datetime.datetime.now()
                            with open('log5.txt', 'a') as f:
                                f.write('\n' + '\t' + str(i + 1) + ' / ' + str(config.trial) + ' : ' + str(now))
                                f.write('\n' + '\t\tCoverage rate = ' + str((learn.coverage[1] - learn.coverage[0])[-1]))

                        if eval('address.' + str(method))['processing'] == 'batch':
                            integrate(data_path=learn.data_path_temp, method=str(method), loss=False, gamma=False, trial=config.trial)        
            
                        elif eval('address.' + str(method))['processing'] == 'online':
                            for loss_temp in config.losses:
                                if str(loss_temp) == 'pinball':
                                    integrate(data_path=learn.data_path_temp, method=str(method), loss=loss_temp, gamma=0, trial=config.trial)
                                else:
                                    for gamma in config.gamma:
                                        integrate_CQR(data_path=learn.data_path_temp, method=str(method), loss=loss_temp, gamma=gamma, trial=config.trial)

                    #CQR追加パッチ
                    #ここ怪しい，config.Iterではないと思う
                    #covlen = str(XX*3/(config.Iter))
                    #with open('log.txt', 'a') as f:
                        #f.write('\n---------------------------------------------')
                        #f.write('\n' + '\t\tCQR Coverage rate = ' + str(covlen))

                            
                    with open('log5.txt', 'a') as f:
                        f.write('\n---------------------------------------------')
                        f.write('END')