#保管庫
data_path_temp = data_path
    data_path ='truth/' + str(noise_type) + '/' + str(outlier_type) + '/outlier_rate=' + str(outlier_rate)  +'/Iter=' + str(config.Iter) + '/trial=' + str(i+1)  
    mkdir(data_path, exist_ok=True)
    data_path = data_path + '/grd_truth.npz'
    np.savez_compressed(data_path, grd_truth)
    data_path = data_path_temp


import graph
import data as dt
import optimize
from integrate import data_integrate as integrate
from algorithms.online.gradient_descent import online_learning as grad

import datetime
import configuration.config as config
import configuration.graph_config as graph_config
import configuration.address as address


import numpy as np

# data 

import graph
import data as dt
import optimize
from integrate import data_integrate as integrate
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

if config.data_flag == 'on':
    for noise_type in config.noise_type_all:
        for outlier_type in config.outlier_type_all:
            for outlier_rate in config.outlier_rate:
                for i in range(config.trial):
                    data_path = 'exp_data/' + 'dim=' + str(config.input_dim) + '/' + str(noise_type) + '/' + str(outlier_type) + '/outlier_rate=' + str(outlier_rate) + '/Iter=' + str(config.Iter) + '/trial=' + str(i + 1)
                    dt.dt(data_path=data_path, Iter=config.Iter, input_dim=config.input_dim, noise_type=noise_type, outlier_type=outlier_type, outlier_rate=outlier_rate)

if config.calib_flag == 'on':
    for noise_type in config.noise_type_all:
    #for noise_type in config.noise_types:
        for outlier_type in config.outlier_type_all:
        #for outlier_type in config.outlier_types:
            for outlier_rate in config.outlier_rate:
                for i in range(config.trial):
                    data_path = 'exp_data/' + 'dim=' + str(config.input_dim) + '/' + str(noise_type) + '/' + str(outlier_type) + '/outlier_rate=' + str(outlier_rate) + '/Iter=' + str(config.Iter) + '/trial=' + str(i + 1)
                    dt.dt_c(data_path=data_path, Iter=config.Iter, input_dim=config.input_dim, noise_type=noise_type, outlier_type=outlier_type, outlier_rate=outlier_rate)

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
                            data_c = np.load(data_path + 'calib/data_c.npz')
                            observation_c = np.load(data_path + 'calib/outlier_c.npz')
                            
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
                outlier_rate = 0.1
                with open('log.txt', 'w') as f:
                    f.write('noise_type : ' + str(noise_type))
                    f.write('\noutlier_type : ' + str(outlier_type))
                    f.write('\noutlier_rate : ' + str(outlier_rate))
                    f.write('\n---------------------------------------------')

                for index_method, method in enumerate(config.methods):
                    with open('log.txt', 'a') as f:
                        f.write('\nMethod : ' + str(method))

                    for index_alpha, alpha in enumerate(alpha_all):
                        with open('log.txt', 'a') as f:
                            f.write('\n---------------------------------------------')
                            f.write('\n' +  str(index_alpha + 1) + ' / ' + str(len(alpha_all)) + ' : ' + str(alpha))
                            f.write('\n---------------------------------------------')
                            
                        for i in range(config.trial):
                            data_path = 'exp_data/' + 'dim=' + str(config.input_dim) + '/' + str(noise_type) + '/' + str(outlier_type) + '/outlier_rate=' + str(outlier_rate) + '/Iter=' + str(config.Iter) + '/trial=' + str(i+1) + '/' 
                            observation = np.load(data_path + 'outlier.npz')
                            noise = np.load(data_path + 'noise.npz')
                            data = np.load(data_path + 'data.npz')
                            data_c = np.load(data_path + 'calib/data_c.npz')
                            observation_c = np.load(data_path + 'calib/outlier_c.npz')
                            
                            if eval('address.' + str(method))['processing'] == 'batch':
                                learn = optimize.batch_learning(observation=observation, noise=noise, data=data, alpha=alpha, method=eval('address.' + str(method)), trial=i+1, outlier_rate=outlier_rate)
                                grd_truth = optimize.gt(data_path=learn.data_path, observation=observation, noise=noise, data=data, alpha=alpha)
                                learn.pre_learning()
                                learn.learning()
                                learn.eval(ground_truth=grd_truth)
                                learn.save()
                                
                            elif eval('address.' + str(method))['processing'] == 'online':
                                learn = optimize.online_learning(observation=observation, noise=noise, data=data, alpha=alpha, method=eval('address.' + str(method)), trial=i+1, outlier_rate=outlier_rate, observation_c=observation_c, data_c=data_c)
                                grd_truth = optimize.gt(data_path=learn.data_path, observation=observation, noise=noise, data=data, alpha=alpha)
                                learn.pre_learning()
                                for loss_temp in config.losses:
                                    if str(loss_temp) == 'pinball':
                                        loss = eval('address.' + str(loss_temp))
                                        loss['gamma'] = 0
                                        learn.learning(loss=loss)
                                        learn.eval(ground_truth=grd_truth)
                                        covlen = learn.CQR(data_c=data_c, observation_c=observation_c)
                                        learn.save()
                                        
                                    else:
                                        for gamma in config.gamma:
                                            loss = eval('address.' + str(loss_temp))
                                            loss['gamma'] = gamma
                                            learn.learning(loss=loss)
                                            learn.eval(ground_truth=grd_truth)
                                            XX = learn.CQR(data_c=data_c, observation_c=observation_c)[0]
                                            learn.save()
                            else:
                                print('ERROR: You should choose a method correctly.')

                            now = datetime.datetime.now()
                            with open('log.txt', 'a') as f:
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
                                        integrate(data_path=learn.data_path_temp, method=str(method), loss=loss_temp, gamma=gamma, trial=config.trial)   

                    #CQR追加パッチ
                    func_est_final = learn.func_est_final.T
                    func_low = func_est_final[:,0]
                    func_high = func_est_final[:,1]
                    scores = learn.CQR(data_c=data_c, observation_c=observation_c)[1]
                    X = np.full([len(func_low), 1], scores)
                    higher = func_high.reshape(-1, 1) + X.reshape(-1, 1)
                    print(X.reshape(-1, 1))
                    print(higher)
                    lower = func_low.reshape(-1, 1) - X.reshape(-1, 1)
                    coverage_h = np.where((higher - learn.output_test > 0), 1, 0)
                    coverage_l = np.where((lower - learn.output_test > 0), 1, 0)
                    XXX = np.sum(coverage_h - coverage_l)
                    covlen = str((XXX + XX)/(2*config.Iter))
                    yobi = str(XXX/config.Iter)
                    saikakunin = str(XX/config.Iter)
                    with open('log.txt', 'a') as f:
                        f.write('\n---------------------------------------------')
                        f.write('\n' + '\t\tCQR Coverage rate = ' + str(covlen))
                        f.write('\n' + '\t\tichiou = ' + str(yobi))
                        f.write('\n' + '\t\tkakunin = ' + str(saikakunin))

                            
                    with open('log.txt', 'a') as f:
                        f.write('\n---------------------------------------------')
                        f.write('END')


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

if config.data_flag == 'on':
    for noise_type in config.noise_type_all:
        for outlier_type in config.outlier_type_all:
            for outlier_rate in config.outlier_rate:
                for i in range(config.trial):
                    data_path = 'exp_data/' + 'dim=' + str(config.input_dim) + '/' + str(noise_type) + '/' + str(outlier_type) + '/outlier_rate=' + str(outlier_rate) + '/Iter=' + str(config.Iter) + '/trial=' + str(i + 1)
                    dt.dt(data_path=data_path, Iter=config.Iter, input_dim=config.input_dim, noise_type=noise_type, outlier_type=outlier_type, outlier_rate=outlier_rate)

if config.calib_flag == 'on':
    for noise_type in config.noise_type_all:
    #for noise_type in config.noise_types:
        for outlier_type in config.outlier_type_all:
        #for outlier_type in config.outlier_types:
            for outlier_rate in config.outlier_rate:
                for i in range(config.trial):
                    data_path = 'exp_data/' + 'dim=' + str(config.input_dim) + '/' + str(noise_type) + '/' + str(outlier_type) + '/outlier_rate=' + str(outlier_rate) + '/Iter=' + str(config.Iter) + '/trial=' + str(i + 1)
                    dt.dt_c(data_path=data_path, Iter=config.Iter, input_dim=config.input_dim, noise_type=noise_type, outlier_type=outlier_type, outlier_rate=outlier_rate)

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
                            data_c = np.load(data_path + 'calib/data_c.npz')
                            observation_c = np.load(data_path + 'calib/outlier_c.npz')
                            
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
                outlier_rate = 0.1
                with open('log.txt', 'w') as f:
                    f.write('noise_type : ' + str(noise_type))
                    f.write('\noutlier_type : ' + str(outlier_type))
                    f.write('\noutlier_rate : ' + str(outlier_rate))
                    f.write('\n---------------------------------------------')

                for index_method, method in enumerate(config.methods):
                    with open('log.txt', 'a') as f:
                        f.write('\nMethod : ' + str(method))

                    for index_alpha, alpha in enumerate(alpha_all):
                        with open('log.txt', 'a') as f:
                            f.write('\n---------------------------------------------')
                            f.write('\n' +  str(index_alpha + 1) + ' / ' + str(len(alpha_all)) + ' : ' + str(alpha))
                            f.write('\n---------------------------------------------')
                            
                        for i in range(config.trial):
                            data_path = 'exp_data/' + 'dim=' + str(config.input_dim) + '/' + str(noise_type) + '/' + str(outlier_type) + '/outlier_rate=' + str(outlier_rate) + '/Iter=' + str(config.Iter) + '/trial=' + str(i+1) + '/' 
                            observation = np.load(data_path + 'outlier.npz')
                            noise = np.load(data_path + 'noise.npz')
                            data = np.load(data_path + 'data.npz')
                            data_c = np.load(data_path + 'calib/data_c.npz')
                            observation_c = np.load(data_path + 'calib/outlier_c.npz')
                            
                            if eval('address.' + str(method))['processing'] == 'batch':
                                learn = optimize.batch_learning(observation=observation, noise=noise, data=data, alpha=alpha, method=eval('address.' + str(method)), trial=i+1, outlier_rate=outlier_rate)
                                grd_truth = optimize.gt(data_path=learn.data_path, observation=observation, noise=noise, data=data, alpha=alpha)
                                learn.pre_learning()
                                learn.learning()
                                learn.eval(ground_truth=grd_truth)
                                learn.save()
                                
                            elif eval('address.' + str(method))['processing'] == 'online':
                                learn = optimize.online_learning(observation=observation, noise=noise, data=data, alpha=alpha, method=eval('address.' + str(method)), trial=i+1, outlier_rate=outlier_rate, observation_c=observation_c, data_c=data_c)
                                grd_truth = optimize.gt(data_path=learn.data_path, observation=observation, noise=noise, data=data, alpha=alpha)
                                learn.pre_learning()
                                for loss_temp in config.losses:
                                    if str(loss_temp) == 'pinball':
                                        loss = eval('address.' + str(loss_temp))
                                        loss['gamma'] = 0
                                        learn.learning(loss=loss)
                                        learn.eval(ground_truth=grd_truth)
                                        covlen = learn.CQR(data_c=data_c, observation_c=observation_c)
                                        learn.save()
                                        
                                    else:
                                        for gamma in config.gamma:
                                            loss = eval('address.' + str(loss_temp))
                                            loss['gamma'] = gamma
                                            learn.learning(loss=loss)
                                            learn.eval(ground_truth=grd_truth)
                                            XX = learn.CQR(data_c=data_c, observation_c=observation_c)[0]
                                            learn.save()
                            else:
                                print('ERROR: You should choose a method correctly.')

                            now = datetime.datetime.now()
                            with open('log.txt', 'a') as f:
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
                                        integrate(data_path=learn.data_path_temp, method=str(method), loss=loss_temp, gamma=gamma, trial=config.trial)   

                    #CQR追加パッチ
                    func_est_final = learn.func_est_final.T
                    func_low = func_est_final[:,0]
                    func_high = func_est_final[:,1]
                    scores = learn.CQR(data_c=data_c, observation_c=observation_c)[1]
                    X = np.full([len(func_low), 1], scores)
                    higher = func_high.reshape(-1, 1) + X.reshape(-1, 1)
                    print(X.reshape(-1, 1))
                    print(higher)
                    lower = func_low.reshape(-1, 1) - X.reshape(-1, 1)
                    coverage_h = np.where((higher - learn.output_test > 0), 1, 0)
                    coverage_l = np.where((lower - learn.output_test > 0), 1, 0)
                    XXX = np.sum(coverage_h - coverage_l)
                    covlen = str((XXX + XX)/(2*config.Iter))
                    yobi = str(XXX/config.Iter)
                    saikakunin = str(XX/config.Iter)
                    with open('log.txt', 'a') as f:
                        f.write('\n---------------------------------------------')
                        f.write('\n' + '\t\tCQR Coverage rate = ' + str(covlen))
                        f.write('\n' + '\t\tichiou = ' + str(yobi))
                        f.write('\n' + '\t\tkakunin = ' + str(saikakunin))

                            
                    with open('log.txt', 'a') as f:
                        f.write('\n---------------------------------------------')
                        f.write('END')
