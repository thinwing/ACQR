import numpy as np
# import matplotlib
# del matplotlib.font_manager.weight_dict['roman']
# matplotlib.font_manager._rebuild()

from graph_package.tool_box.pdf2png import convert as cv
import matplotlib.pyplot as plt
from os import makedirs as mkdir

import sys
sys.path.append('../')
from integrate import get_path
from configuration import graph_config as grp
from configuration import config

        
def comp_proposed_coverage(data_path, loss_list, method_list): 
    method_list_flatten = method_list.flatten()
    list_flatten = loss_list.flatten()
    iteration = np.arange(config.Iter)
    range_alpha = config.range_alpha[config.start:config.limit]
    
        
    for _, range_alpha_temp in enumerate(range_alpha):
        ground_alpha = np.ones(config.Iter) * range_alpha_temp
        fig = plt.figure(figsize=(12, 8))
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams['text.usetex'] = True
        ax = fig.add_subplot(1, 1, 1)
        data_path_alpha = data_path + '/alpha=' + str(range_alpha_temp)

        for index_method, method in enumerate(method_list_flatten):                
            for _, item in enumerate(list_flatten):
                if method == 'multi_kernel' and item == 'pinball':
                    continue

                data_path_detail, _ = get_path(data_path=data_path_alpha, method=method, loss=item, gamma=eval("grp." + str(item))['gamma'])
                            
                method_temp = np.load(data_path_detail)
                coverage_temp = method_temp['coverage']
                coverage = coverage_temp[1] - coverage_temp[0]
                
                ax.plot(iteration, coverage, label=eval('grp.' + str(item))['loss_name'] + eval('grp.' + str(method))['fig_name2'], color=eval('grp.' + str(item))['color'][index_method], linewidth=grp.linewidth)
        # ax.axvline(x=config.incident[1], linestyle=':', linewidth=4)


        ax.plot(iteration, ground_alpha, label='ground truth', color='black', linewidth=grp.linewidth, linestyle='dashdot')


        ax.set_ylabel('Actual coverage rate', fontsize=grp.font_size)
        ax.set_xlabel('Iteration', fontsize=grp.font_size)
        # ax.set_xscale('log')
        ax.grid()
        ax.legend(fontsize=32)
        ax.set_ylim(0, 1)
        plt.tick_params(labelsize=grp.ticks)

        save_path_alpha = data_path_alpha.replace('text','graph')
        mkdir(save_path_alpha, exist_ok=True)
        save_name = save_path_alpha + '/coverage_gamma.pdf'
        plt.savefig(save_name, bbox_inches='tight')        
        plt.clf()
        plt.close()
        image_name = save_name.replace('pdf', 'png')
        cv(save_name, image_name)

def comp_proposed_coverage_db(data_path, loss_list, method_list): 
    method_list_flatten = method_list.flatten()
    list_flatten = loss_list.flatten()
    iteration = np.arange(config.Iter)
    range_alpha = config.range_alpha[config.start:config.limit]
    
        
    for _, range_alpha_temp in enumerate(range_alpha):
        fig = plt.figure(figsize=(12, 8))
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams['text.usetex'] = True
        ax = fig.add_subplot(1, 1, 1)
        data_path_alpha = data_path + '/alpha=' + str(range_alpha_temp)

        for index_method, method in enumerate(method_list_flatten):                
            for _, item in enumerate(list_flatten):
                if method == 'multi_kernel' and item == 'pinball':
                    continue

                data_path_detail, _ = get_path(data_path=data_path_alpha, method=method, loss=item, gamma=eval("grp." + str(item))['gamma'])
                            
                method_temp = np.load(data_path_detail)
                coverage_temp = method_temp['coverage']
                coverage = coverage_temp[1] - coverage_temp[0]
                
                coverage_db = 10 * np.log((coverage - range_alpha_temp) ** 2 / range_alpha_temp ** 2 + 0.001)
                
                ax.plot(iteration, coverage_db, label=eval('grp.' + str(item))['loss_name'] + eval('grp.' + str(method))['fig_name2'], color=eval('grp.' + str(item))['color'][index_method], linewidth=grp.linewidth)
        # ax.axvline(x=config.incident[1], linestyle=':', linewidth=6)


        ax.set_ylabel('Coverage rate error', fontsize=grp.font_size)
        ax.set_xlabel('Iteration', fontsize=grp.font_size)
        # ax.set_xscale('log')
        ax.grid()
        ax.legend(fontsize=36)
        # ax.set_ylim(0, 1)
        plt.tick_params(labelsize=grp.ticks)
        
        save_path_alpha = data_path_alpha.replace('text','graph')
        mkdir(save_path_alpha, exist_ok=True)
        save_name = save_path_alpha + '/coverage_gamma_db.pdf'
        plt.savefig(save_name, bbox_inches='tight')        
        plt.clf()
        plt.close()

        image_name = save_name.replace('pdf', 'png')
        cv(save_name, image_name)