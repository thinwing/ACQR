import graph
import data as dt
import optimize
import optimize_CQR100
from integrate import data_integrate as integrate
from integrate import data_integrate_CQR100 as integrate_CQR100
from algorithms.online.gradient_descent import online_learning as grad

import datetime
import configuration.config as config
import configuration.graph_config as graph_config
import configuration.address as address


import numpy as np

# data 

alpha_all = np.array([[0.05, 1.00]])

# num = config.num_divide
# if num > 0:
#     num_len = int(len(alpha_all_temp) / 3)
#     if num < 3:
#         alpha_all = alpha_all_temp[(num - 1) * num_len : num * num_len]
#     else:
#         alpha_all = alpha_all_temp[(num - 1) * num_len :]
# else:
#     alpha_all = alpha_all_temp
            
for noise_type in config.noise_types:
    for outlier_type in config.outlier_types:
        #for outlier_rate in config.outlier_rate:
            outlier_rate = 0.04
            for index_method, method in enumerate(config.methods):
                for index_alpha, alpha in enumerate(alpha_all):
                    data_path = 'result/text/dim=1/linear_expansion/sparse/outlier_rate=0.04/Iter=1000/alpha=0.95'
                    if eval('address.' + str(method))['processing'] == 'batch':
                        integrate(data_path=data_path, method=str(method), loss=False, gamma=False, trial=config.trial)        
        
                    elif eval('address.' + str(method))['processing'] == 'online':
                        for loss_temp in config.losses:
                            if str(loss_temp) == 'pinball':
                                integrate(data_path=data_path, method=str(method), loss=loss_temp, gamma=0, trial=config.trial)
                            else:
                                for gamma in config.gamma:
                                    integrate_CQR100(data_path=data_path, method=str(method), loss=loss_temp, gamma=gamma, trial=config.trial)

                #CQR追加パッチ
                #ここ怪しい，config.Iterではないと思う
                #covlen = str(XX*3/(config.Iter))
                #with open('log.txt', 'a') as f:
                    #f.write('\n---------------------------------------------')
                    #f.write('\n' + '\t\tCQR Coverage rate = ' + str(covlen))