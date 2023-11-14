import numpy as np
import matplotlib.pyplot as plt
from os import makedirs as mkdir
from graph_package.tool_box.pdf2png import convert as cv
import sys
sys.path.append('../')
from configuration import graph_config as grp
from configuration import config
from integrate import get_path



def gamma_coverage(data_path, loss_list, method='single_kernel', alpha=config.alpha_range):
    gamma = config.gamma

    # gamma_base = np.round((np.arange(11)) * 0.1, 2)
    gamma_base = gamma

    # tem = int(np.amin(result_coverage_db))
    
    fig = plt.figure(figsize=(12, 8))
    plt.tick_params(labelsize=grp.ticks)
    plt.rcParams['font.family'] = 'Times New Roman'    
    plt.rcParams['text.usetex'] = True
    ax1 = fig.add_subplot(1, 1, 1)
        
    list_flatten = loss_list.flatten()
    
    result_coverage = np.zeros(len(gamma))
    
    # pinball loss
    data_path_alpha = data_path + '/alpha=' + str(alpha)
    
    for _, item in enumerate(list_flatten):
        if item == 'pinball':
            data_path_detail, _ = get_path(data_path=data_path_alpha, method=method, loss='pinball', gamma=0)
            method_result = np.load(data_path_detail)    
            coverage = (method_result['coverage'][1] - method_result['coverage'][0]).reshape(-1)
            result_coverage_pinball = coverage[-1] * np.ones(len(gamma))
            
            ax1.plot(gamma, result_coverage_pinball, label=eval('grp.' + str(method))['fig_name'] + ', loss : ' + eval('grp.' + 'pinball')['loss_name'], color=eval('grp.' + 'pinball')['color'], linewidth=grp.linewidth, marker=eval('grp.' + 'pinball')['marker'], markersize=grp.marker_size)
        else:
            for index_gamma, gamma_temp in enumerate(gamma):
                # data load
                data_path_detail, _ = get_path(data_path=data_path_alpha, method=method, loss=item, gamma=gamma_temp)
                
                method_result = np.load(data_path_detail)
                coverage = (method_result['coverage'][1] - method_result['coverage'][0]).reshape(-1)
                
                result_coverage[index_gamma] = coverage[-1]
        
            ax1.plot(gamma, result_coverage, label=eval('grp.' + str(method))['fig_name'] + ', loss : ' + eval('grp.' + str(item))['loss_name'], color=eval('grp.' + str(item))['color'][0], linewidth=grp.linewidth, marker=eval('grp.' + str(item))['marker'], markersize=grp.marker_size)
    
    # ax1.set_title('Range of r = ' + str(alpha) + ' and coverage by \u03b3', fontsize=grp.font_size)
    ax1.set_ylabel('Actual coverage rate', fontsize=grp.font_size)
    ax1.set_xlabel(r'$\gamma$', fontsize=grp.font_size)
    ax1.set_ylim(0, 1)
    ax1.set_xticks(gamma_base)
    ax1.grid()
    ax1.legend(fontsize=grp.font_size)
    
    
    save_path = data_path.replace('text', 'graph')
    mkdir(save_path, exist_ok=True)
    save_name = save_path + '/coverage_gamma.pdf'
    plt.savefig(save_name, bbox_inches='tight')
    plt.clf()
    plt.close()
    image_name = save_name.replace('pdf', 'png')
    cv(save_name, image_name)
    
    
def gamma_error(data_path, loss_list, method='single_kernel', alpha=config.alpha_range):
    gamma = config.gamma

    gamma_base = np.round((np.arange(11)) * 0.1, 2)
    
    list_flatten = loss_list.flatten()
    
    # tem = int(np.amin(result_coverage_db))
    for i in range(3):
        fig = plt.figure(figsize=(12, 8))
        ax1 = fig.add_subplot(1, 1, 1)    
        plt.tick_params(labelsize=grp.ticks)
        plt.rcParams['font.family'] = 'Times New Roman'    
        plt.rcParams['text.usetex'] = True
    
        result_coverage = np.zeros(len(gamma))
        # pinball loss
        data_path_alpha = data_path + '/alpha=' + str(alpha)
        
    
        for _, item in enumerate(list_flatten):
            if item == 'pinball':
                data_path_detail, _ = get_path(data_path=data_path_alpha, method=method, loss='pinball', gamma=0)
                method_result = np.load(data_path_detail)    
                coverage = method_result['coverage_db'][i]
                result_coverage_pinball = coverage[-1] * np.ones(len(gamma))
                
                ax1.plot(gamma, result_coverage_pinball, label=eval('grp.' + 'pinball')['loss_name'], color=eval('grp.' + 'pinball')['color'], linewidth=grp.linewidth, marker=eval('grp.' + 'pinball')['marker'], markersize=grp.marker_size)

            else:
                for index_gamma, gamma_temp in enumerate(gamma):
                    # data load
                    data_path_detail, _ = get_path(data_path=data_path_alpha, method=method, loss=item, gamma=gamma_temp)
                    
                    method_result = np.load(data_path_detail)
                    coverage = method_result['coverage_db'][i]
                    
                    result_coverage[index_gamma] = coverage[-1]
                
                ax1.plot(gamma, result_coverage, label=eval('grp.' + str(item))['loss_name'], color=eval('grp.' + str(item))['color'][0], linewidth=grp.linewidth, marker=eval('grp.' + str(item))['marker'], markersize=grp.marker_size)
        
        ax1.set_title('Error between ground truth and estimate result', fontsize=grp.font_size)
        ax1.set_ylabel('Error', fontsize=grp.font_size)
        ax1.set_xlabel(r'$\gamma$', fontsize=grp.font_size)
        ax1.set_xticks(gamma_base)
        ax1.grid()
        ax1.legend(fontsize=grp.font_size)
        
        save_path = data_path.replace('text', 'graph')
        mkdir(save_path, exist_ok=True)
        save_name = save_path + '/' + grp.title_coverage_db_gamma[i]
        plt.savefig(save_name, bbox_inches='tight')
        plt.clf()
        plt.close()

        image_name = save_name.replace('pdf', 'png')
        cv(save_name, image_name)