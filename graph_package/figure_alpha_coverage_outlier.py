import numpy as np
# import matplotlib
# del matplotlib.font_manager.weight_dict['roman']
# matplotlib.font_manager._rebuild()

import matplotlib.pyplot as plt
from os import makedirs as mkdir
from graph_package.tool_box.pdf2png import convert as cv

import sys
sys.path.append('../')
from configuration import graph_config as grp
from configuration import config
from integrate import get_path

def outlier_coverage(data_path, list, loss=False, gamma=False):
    range_alpha = grp.range_alpha[0]   
    # tem = int(np.amin(result_coverage_db))
    
    fig = plt.figure(figsize=(16, 12))
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['text.usetex'] = True
    ax1 = fig.add_subplot(1, 1, 1)
    
    list_flatten = (list.T).flatten()
    
    result_coverage = np.zeros(len(config.outlier_rate))
    width_base = config.outlier_rate[0] / len(list_flatten)
    width = width_base * 0.8 
    x_ticks = config.outlier_rate
    result_coverage_interval = np.zeros([2, len(config.outlier_rate)])
    
    for index_method, item in enumerate(list_flatten):
        data_path_alpha = data_path + '/alpha=' + str(range_alpha)
        for index_outlier, _ in enumerate(config.outlier_rate):
            # data load
            
            if index_outlier > 0:
                data_path_alpha = data_path_alpha.replace('outlier_rate=' + str(config.outlier_rate[index_outlier - 1]), 'outlier_rate=' + str(config.outlier_rate[index_outlier]))

            data_path_detail, _ = get_path(data_path=data_path_alpha, method=item, loss=loss, gamma=gamma)
            method = np.load(data_path_detail)
            # print(data_path_detail)
            coverage = (method['coverage'][1] - method['coverage'][0]).reshape(-1)
            
            result_coverage[index_outlier] = coverage[-1] / range_alpha

            result_coverage_interval[:,index_outlier] = method['coverage_interval'] / range_alpha
        
        temp = 20
        result_coverage_interval = abs(result_coverage_interval - result_coverage)
        # ax1.errorbar(x_ticks + ((width + width / temp) * (index_method - 1.5)), result_coverage, yerr=result_coverage_interval, capsize=24, fmt=eval('grp.' + str(item))['marker'], label=eval('grp.' + str(item))['fig_name'], ecolor=eval('grp.' + str(item))['color'], elinewidth=6, markersize=12, color='black')
        ax1.errorbar(x_ticks, result_coverage, yerr=result_coverage_interval, capsize=24, fmt=eval('grp.' + str(item))['marker'],  ecolor=eval('grp.' + str(item))['color'], elinewidth=6, markersize=12, color='black')
        ax1.plot(x_ticks,result_coverage, label=eval('grp.' + str(item))['fig_name'], color=eval('grp.' + str(item))['color'], linewidth=4)
        # ax1.bar(x_ticks + ((width + width / temp) * (index_method - 1.5)), result_coverage, label=eval('grp.' + str(item))['fig_name'], color=eval('grp.' + str(item))['color'], width=width, align='center')
        
    # ax1.set_title('Range of \u03b1 and coverage', fontsize=grp.font_size)
    ax1.set_ylabel(r'Actual coverage rate / $\alpha$', fontsize=grp.font_size)
    ax1.set_xlabel(r'Outlier rate', fontsize=grp.font_size)
    # ax1.set_ylim(1.0, 1.1)
    ax1.set_xticks(x_ticks)
    # ax1.set_yticks(range_alpha)
    plt.tick_params(labelsize=grp.ticks)

    ax1.grid()
    ax1.legend(fontsize=30)    
    
    save_path = data_path.replace('text', 'graph')
    mkdir(save_path, exist_ok=True)
    save_name = save_path + '/' + str(range_alpha) + 'coverage_outlier.pdf'
    plt.savefig(save_name, bbox_inches='tight')
    plt.clf()
    plt.close()

    image_name = save_name.replace('pdf', 'png')
    cv(save_name, image_name)


    
def outlier_error(data_path, list, loss=False, gamma=False):
    range_alpha = grp.range_alpha[0]   
    list_flatten = (list.T).flatten()

    result_coverage = np.zeros(len(config.outlier_rate))

    width_base = config.outlier_rate[0] / len(list_flatten)
    width = width_base * 0.8 
    x_ticks = config.outlier_rate
    result_coverage_interval = np.zeros([2, len(config.outlier_rate)])
        
    for i in range(3):
        fig = plt.figure(figsize=(16, 12))
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams['text.usetex'] = True
        ax2 = fig.add_subplot(1, 1, 1)
        for index_method, item in enumerate(list_flatten):
            data_path_alpha = data_path + '/alpha=' + str(range_alpha)
            for index_outlier, _ in enumerate(config.outlier_rate):
                # data load
                
                if index_outlier > 0:
                    data_path_alpha = data_path_alpha.replace('outlier_rate=' + str(config.outlier_rate[index_outlier - 1]), 'outlier_rate=' + str(config.outlier_rate[index_outlier]))
                    
                data_path_detail, _ = get_path(data_path=data_path_alpha, method=item, loss=loss, gamma=gamma)
                
                method = np.load(data_path_detail)
                coverage = method['coverage_db'][i]
                
                result_coverage[index_outlier] = coverage[-1]
                result_coverage_interval[:, index_outlier] = abs(method['coverage_db_interval'][i] - coverage[-1])

            temp = 20
            # ax2.errorbar(x_ticks + ((width + width / temp) * (index_method - 1.5)), result_coverage, yerr=result_coverage_interval, label=eval('grp.' + str(item))['fig_name'], capsize=24, fmt=eval('grp.' + str(item))['marker'], ecolor=eval('grp.' + str(item))['color'], elinewidth=6, markersize=12, color='black')
            ax2.errorbar(x_ticks, result_coverage, yerr=result_coverage_interval, capsize=24, fmt=eval('grp.' + str(item))['marker'], ecolor=eval('grp.' + str(item))['color'], elinewidth=6, markersize=12, color='black')
            ax2.plot(x_ticks, result_coverage, label=eval('grp.' + str(item))['fig_name'], color=eval('grp.' + str(item))['color'], linewidth=4)
            # ax2.bar(x_ticks + ((width + width / temp) * (index_method - 1.5)), result_coverage, label=eval('grp.' + str(item))['fig_name'], color=eval('grp.' + str(item))['color'], width=width, align='center')
            
        ax2.set_ylabel(grp.title_db[i], fontsize=grp.font_size)
        ax2.set_xlabel(r'Outlier rate', fontsize=grp.font_size)

        ax2.set_xticks(x_ticks)
        ax2.grid()
        ax2.legend(fontsize=30)
        plt.tick_params(labelsize=grp.ticks)
        save_path = data_path.replace('text', 'graph')
        mkdir(save_path, exist_ok=True)
        save_name = save_path + '/' + str(range_alpha) + grp.title_outlier_coverage_db[i]
        plt.savefig(save_name, bbox_inches='tight')

        plt.clf()
        plt.close()

        image_name = save_name.replace('pdf', 'png')
        cv(save_name, image_name)
