from graph_package import *
import numpy as np
from configuration import config
from configuration import address
from configuration import graph_config as grp



class _graph():
    def __init__(self, input_dim=config.input_dim, noise_type=config.noise_type, outlier_type=config.outlier_type, outlier_rate=config.outlier_rate_single, Iter=config.Iter):
        self.data_path = 'result/text/dim=' + str(input_dim) + '/' + str(noise_type) + '/' + str(outlier_type) +'/outlier_rate=' + str(outlier_rate) +  '/Iter=' + str(Iter)
        print(self.data_path) 

#ここ変えた
    def bo_figure(self, list, alpha, trial=1):
        # batch and online
        for trial_temp in range(trial):
            data_path = self.data_path + '/alpha=' + str(alpha) + '/trial=' + str(trial_temp + 1)
            #fig(data_path=data_path, list=list)

            #fig_range(data_path=data_path, list=list)

            figACI(data_path=data_path, trial = trial_temp, list=list)
            fig_rangeACI(data_path=data_path, trial = trial_temp, list=list)

            #figCQR(data_path=data_path, trial = trial_temp, list=list)
            #fig_rangeCQR(data_path=data_path, trial = trial_temp, list=list)    
        
    def bo_coverage(self, list):
        alpha_coverage(data_path=self.data_path, list=list)
        alpha_error(data_path=self.data_path, list=list)   
        
    def o_coverage(self, list, loss=grp.loss_base, gamma=eval('grp.' + str(grp.loss_base))['gamma']):
        comp_proposed_coverage(data_path=self.data_path, list=list, loss=loss, gamma=gamma)
        #comp_proposed_coverage_both(data_path=self.data_path, list=list, loss=loss, gamma=gamma)
        
        #comp_proposed_coverage_db(data_path=self.data_path, list=list, loss=loss, gamma=gamma)
        #comp_proposed_range_ave(data_path=self.data_path, list=list, loss=loss, gamma=gamma)
        
    def o_gamma_coverage(self, loss_list, alpha, method='single_kernel'):
        gamma_coverage(data_path=self.data_path, loss_list=loss_list, method='single_kernel', alpha=alpha)
        gamma_error(data_path=self.data_path, loss_list=loss_list, method='single_kernel', alpha=alpha)
        
        #gamma_coverage3(data_path=self.data_path, loss_list=loss_list, method='single_kernel', alpha=alpha)
        #gamma_error3(data_path=self.data_path, loss_list=loss_list, method='single_kernel', alpha=alpha)
        
        # comp_proposed_coverage_gamma(data_path=self.data_path, loss_list=loss_list, method_list=method)
        # comp_proposed_coverage_gamma_db(data_path=self.data_path, loss_list=loss_list, method_list=method)
        # comp_proposed_coverage_db_gamma(data_path=self.data_path, loss_list=loss_list, method_list=method)
    
    def outlier_coverage(self, list):
        data_path = self.data_path.replace('outlier_rate=' + str(config.outlier_rate_single), 'outlier_rate=' + str(config.outlier_rate[0]))
        out_cov(data_path=data_path, list=list)
        out_err(data_path=data_path, list=list)

    
if __name__ == '__main__':
    gr = _graph()
    #gr.bo_figure(list=grp.list_graph, alpha=0.95)

    #gr.bo_coverage(list=grp.list_graph_coverage)    
    #for index, item in enumerate(grp.loss_list):
        #gr.o_coverage(list=grp.list_graph_online, loss=item, gamma=eval('grp.' + str(item))['gamma'])
        #print(index + 1)

    #ここフォントの問題あり    
    gr.o_gamma_coverage(loss_list=grp.loss_list, alpha=config.alpha_range, method=grp.list_graph_online)
    
    #gr.outlier_coverage(list=grp.list_graph_coverage)