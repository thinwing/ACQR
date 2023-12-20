from graph_package import *

from configuration import config
from configuration import address
from configuration import graph_config as grps
import matplotlib.pyplot as plt
from os import makedirs as mkdir
import numpy as np
from graph_package.tool_box.pdf2png import convert as cv

#for noise_type in config.noise_type_all:
    #for outlier_type in config.outlier_type_all:
        #for outlier_rate in config.outlier_rate:
            #for method in config.method_all:
                #for alpha in config.alpha_all:
                #for i in range(config.trial):
gamma = config.gamma
trial = config.trial

result_coverage = np.zeros((len(gamma), trial))
result_coverage_m = np.zeros((len(gamma), trial))


for noise_type in config.noise_types:
    for outlier_type in config.outlier_types:
        outlier_rate = 0.04
        for method in config.methods:
            for index_gamma, gamma_temp in enumerate(gamma):
                for i in range(config.trial):
                    # data load
                    data_path_detail = 'result/text/dim=1/linear_expansion/sparse/outlier_rate=0.04/Iter=1000/alpha=0.95/trial=' + str(i+1) + '/online/pinball_moreau/' + '/\u03b3=' + str(gamma_temp) + '/CQR/' + method + '.npz'
                    #'data_path_detail, _ = get_path_CQR(data_path=data_path_alpha, method=method, loss=item, gamma=gamma_temp)'
                    method_result = np.load(data_path_detail)
                    coverage = (method_result['coverage'][1] - method_result['coverage'][0]).reshape(-1)
                    if method == 'single_kernel':
                        result_coverage[index_gamma, i] = coverage[-1]
                    else:
                        result_coverage_m[index_gamma, i] = coverage[-1]

for noise_type in config.noise_types:
    for outlier_type in config.outlier_types:
        outlier_rate = 0.04
        for method in config.methods:
            for index_gamma, gamma_temp in enumerate(gamma):
                data_path = 'result/text/dim=1/linear_expansion/sparse/outlier_rate=0.04/Iter=1000/alpha=0.95/online/pinball_moreau/' + '/\u03b3=' + str(gamma_temp) + '/CQR'
                x = result_coverage[index_gamma, :]
                
                plt.hist(x)
                plt.title('\u03b3=' + str(gamma_temp) + 'single', fontsize=20)  # (3) タイトル
                plt.xlabel('Coveragerate', fontsize=20)            # (4) x軸ラベル
                plt.ylabel('Frequency', fontsize=20)
                
                save_path = data_path.replace('text', 'graph')
                #mkdir(save_path, exist_ok=True)
                save_name = save_path + '/hist.pdf'
                plt.savefig(save_name, bbox_inches='tight')
                plt.clf()
                plt.close()
                image_name = save_name.replace('pdf', 'png')
                np.savez_compressed(data_path, coverage=self.coverage, coverage_all=self.coverage_all, range_ave=self.range_func_est_ave, coverage_db=self.coverage_db, func_est=self.func_est_final, input=self.input)
                cv(save_name, image_name)

for noise_type in config.noise_types:
    for outlier_type in config.outlier_types:
        outlier_rate = 0.04
        for index_gamma, gamma_temp in enumerate(gamma):
            data_path = 'result/text/dim=1/linear_expansion/sparse/outlier_rate=0.04/Iter=1000/alpha=0.95/online/pinball_moreau' + '/\u03b3=' + str(gamma_temp) + '/CQR'
            x = result_coverage_m[index_gamma, :]
            
            plt.hist(x)
            plt.title('\u03b3=' + str(gamma_temp) + 'multi', fontsize=20)  # (3) タイトル
            plt.xlabel('Coveragerate', fontsize=20)            # (4) x軸ラベル
            plt.ylabel('Frequency', fontsize=20)

            save_path = data_path.replace('text', 'graph')
            #mkdir(save_path, exist_ok=True)
            save_name = save_path + '/hist_m.pdf'
            plt.savefig(save_name, bbox_inches='tight')
            plt.clf()
            plt.close()
            image_name = save_name.replace('pdf', 'png')
            cv(save_name, image_name)