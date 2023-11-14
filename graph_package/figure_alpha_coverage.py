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



# def alpha_coverage(data_path, list, loss=False, gamma=False):
#     range_alpha = config.range_alpha

#     range_alpha_base = np.round((np.arange(11)) * 0.1, 2)
    
#     y = range_alpha
#     # tem = int(np.amin(result_coverage_db))
    
#     fig = plt.figure(figsize=(12, 11))
#     plt.rcParams['font.family'] = 'Times New Roman'
#     plt.rcParams['text.usetex'] = True
#     ax1 = fig.add_subplot(1, 1, 1)
    
#     ax1.plot(range_alpha, y, label='ground Truth', color='black', linewidth=4, linestyle='dashed')

    
#     list_flatten = (list.T).flatten()
    
#     result_coverage = np.zeros(len(config.range_alpha))
    
#     for _, item in enumerate(list_flatten):
#         for index_alpha, range_alpha_temp in enumerate(range_alpha):
#             # data load
#             data_path_alpha = data_path + '/alpha=' + str(range_alpha_temp)
#             data_path_detail, _ = get_path(data_path=data_path_alpha, method=item, loss=loss, gamma=gamma)
            
#             method = np.load(data_path_detail)
#             coverage = (method['coverage'][1] - method['coverage'][0]).reshape(-1)
            
#             result_coverage[index_alpha] = coverage[-1]
        
#         ax1.plot(range_alpha, result_coverage, label=eval('grp.' + str(item))['fig_name'], color=eval('grp.' + str(item))['color'], linewidth=grp.linewidth, marker=eval('grp.' + str(item))['marker'], markersize=grp.marker_size)
    
#     # ax1.set_title('Range of \u03b1 and coverage', fontsize=grp.font_size)
#     ax1.set_ylabel('Actual coverage rate', fontsize=grp.font_size)
#     ax1.set_xlabel(r'Prespecified coverage rate $\alpha$', fontsize=grp.font_size)
#     ax1.set_ylim(0, 1)
#     ax1.set_xticks(range_alpha_base)
#     ax1.set_yticks(range_alpha_base)
#     plt.tick_params(labelsize=grp.ticks)

#     ax1.grid()
#     ax1.legend(fontsize=30)    
    
#     save_path = data_path.replace('text', 'graph')
#     mkdir(save_path, exist_ok=True)
#     plt.savefig(save_path + '/coverage_alpha.pdf', bbox_inches='tight')
#     plt.clf()
#     plt.close()

#     fig = plt.figure(figsize=(12, 11))
#     plt.rcParams['font.family'] = 'Times New Roman'
#     plt.rcParams['text.usetex'] = True
#     ax1 = fig.add_subplot(1, 1, 1)
    
#     y = np.ones(len(range_alpha))
#     ax1.plot(range_alpha, y, label='ground Truth', color='black', linewidth=4, linestyle='dashed')

#     list_flatten = (list.T).flatten()
    
#     result_coverage = np.zeros(len(config.range_alpha))
    
#     for _, item in enumerate(list_flatten):
#         for index_alpha, range_alpha_temp in enumerate(range_alpha):
#             # data load
#             data_path_alpha = data_path + '/alpha=' + str(range_alpha_temp)
#             data_path_detail, _ = get_path(data_path=data_path_alpha, method=item, loss=loss, gamma=gamma)
            
#             method = np.load(data_path_detail)
#             coverage = (method['coverage'][1] - method['coverage'][0]).reshape(-1)
            
#             result_coverage[index_alpha] = coverage[-1]
        
#         result_coverage_rate = result_coverage / range_alpha
#         ax1.plot(range_alpha, result_coverage_rate, label=eval('grp.' + str(item))['fig_name'], color=eval('grp.' + str(item))['color'], linewidth=grp.linewidth, marker=eval('grp.' + str(item))['marker'], markersize=grp.marker_size)
    
#     # ax1.set_title('Range of \u03b1 and coverage', fontsize=grp.font_size)
#     ax1.set_ylabel('Actual coverage rate / Prespecified coverage rate', fontsize=grp.font_size)
#     ax1.set_xlabel(r'Prespecified coverage rate $\alpha$', fontsize=grp.font_size)
#     ax1.set_xticks(range_alpha_base)
#     plt.tick_params(labelsize=grp.ticks)

#     ax1.grid()
#     ax1.legend(fontsize=30)    
    
#     save_path = data_path.replace('text', 'graph')
#     mkdir(save_path, exist_ok=True)
#     plt.savefig(save_path + '/coverage_alpha_rate.pdf', bbox_inches='tight')
#     plt.clf()
#     plt.close()

def alpha_coverage(data_path, list, loss=False, gamma=False):
    range_alpha = config.range_alpha[config.start:config.limit]    
    # tem = int(np.amin(result_coverage_db))
    
    fig = plt.figure(figsize=(12, 9))
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['text.usetex'] = True
    ax1 = fig.add_subplot(1, 1, 1)
    
    list_flatten = (list.T).flatten()
    
    result_coverage = np.zeros(len(range_alpha))
    width_base = 0.05 / len(list_flatten)
    width = width_base * 0.8 
    x_ticks = range_alpha
    result_coverage_interval = np.zeros([2, len(range_alpha)])
    
    for index_method, item in enumerate(list_flatten):
        for index_alpha, range_alpha_temp in enumerate(range_alpha):
            # data load
            data_path_alpha = data_path + '/alpha=' + str(range_alpha_temp)
            data_path_detail, _ = get_path(data_path=data_path_alpha, method=item, loss=loss, gamma=gamma)
            method = np.load(data_path_detail)
            # print(data_path_detail)
            coverage = (method['coverage'][1] - method['coverage'][0]).reshape(-1)
            
            result_coverage[index_alpha] = coverage[-1] / range_alpha_temp

            result_coverage_interval[:,index_alpha] = method['coverage_interval'] / range_alpha_temp
        
        temp = 20
        result_coverage_interval = abs(result_coverage_interval - result_coverage)
        ax1.errorbar(x_ticks + ((width + width / temp) * (index_method - 1.5)), result_coverage, yerr=result_coverage_interval, capsize=24, fmt=eval('grp.' + str(item))['marker'], label=eval('grp.' + str(item))['fig_name'], ecolor=eval('grp.' + str(item))['color'], elinewidth=6, markersize=12, color='black')
        # ax1.bar(x_ticks + ((width + width / temp) * (index_method - 1.5)), result_coverage, label=eval('grp.' + str(item))['fig_name'], color=eval('grp.' + str(item))['color'], width=width, align='center')
        
    # ax1.set_title('Range of \u03b1 and coverage', fontsize=grp.font_size)
    ax1.set_ylabel(r'Actual coverage rate / $\alpha$', fontsize=grp.font_size)
    ax1.set_xlabel(r'Prespecified coverage rate $\alpha$', fontsize=grp.font_size)
    ax1.set_ylim(1, 1.1)
    ax1.set_xticks(range_alpha)
    # ax1.set_yticks(range_alpha)
    plt.tick_params(labelsize=grp.ticks)

    ax1.grid(axis='y')
    ax1.legend(fontsize=30)    
    
    save_path = data_path.replace('text', 'graph')
    mkdir(save_path, exist_ok=True)
    save_name = save_path + '/' + str(range_alpha) + 'coverage_alpha.pdf'
    plt.savefig(save_name, bbox_inches='tight')
    plt.clf()
    plt.close()

    image_name = save_name.replace('pdf', 'png')
    cv(save_name, image_name)

# def alpha_error(data_path, list, loss=False, gamma=False):
#     range_alpha = config.range_alpha
#     list_flatten = (list.T).flatten()

#     range_alpha_base = np.round((np.arange(11)) * 0.1, 2)

#     result_coverage = np.zeros(len(config.range_alpha))
        
#     for i in range(3):
#         fig = plt.figure(figsize=(12, 8))
#         plt.rcParams['font.family'] = 'Times New Roman'
#         plt.rcParams['text.usetex'] = True
#         ax2 = fig.add_subplot(1, 1, 1)
#         for _, item in enumerate(list_flatten):
#             for index_alpha, range_alpha_temp in enumerate(range_alpha):
#                 # data load
#                 data_path_alpha = data_path + '/alpha=' + str(range_alpha_temp)
#                 data_path_detail, _ = get_path(data_path=data_path_alpha, method=item, loss=loss, gamma=gamma)
                
#                 method = np.load(data_path_detail)
#                 coverage = method['coverage_db'][i]
                
#                 result_coverage[index_alpha] = coverage[-1]
            
#             ax2.plot(range_alpha, result_coverage, label=eval('grp.' + str(item))['fig_name'], color=eval('grp.' + str(item))['color'], linewidth=grp.linewidth, marker=eval('grp.' + str(item))['marker'], markersize=grp.marker_size)
    
#         ax2.set_ylabel(grp.title_db[i], fontsize=grp.font_size)
#         ax2.set_xlabel(r'Prespecified coverage rate $\alpha$', fontsize=grp.font_size)
    
#         ax2.set_xticks(range_alpha_base)
#         ax2.grid()
#         ax2.legend(fontsize=30)
#         plt.tick_params(labelsize=grp.ticks)
#         save_path = data_path.replace('text', 'graph')
#         mkdir(save_path, exist_ok=True)
#         plt.savefig(save_path + '/' + grp.title_coverage_db[i], bbox_inches='tight')

#         plt.clf()
#         plt.close()
    
def alpha_error(data_path, list, loss=False, gamma=False):
    range_alpha = config.range_alpha[config.start:config.limit]    
    list_flatten = (list.T).flatten()

    result_coverage = np.zeros(len(range_alpha))

    width_base = 0.05 / len(list_flatten)
    width = width_base * 0.8 
    x_ticks = range_alpha
    result_coverage_interval = np.zeros([2, len(range_alpha)])
        
    for i in range(3):
        fig = plt.figure(figsize=(12, 9))
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams['text.usetex'] = True
        ax2 = fig.add_subplot(1, 1, 1)
        for index_method, item in enumerate(list_flatten):
            for index_alpha, range_alpha_temp in enumerate(range_alpha):
                # data load
                data_path_alpha = data_path + '/alpha=' + str(range_alpha_temp)
                data_path_detail, _ = get_path(data_path=data_path_alpha, method=item, loss=loss, gamma=gamma)
                
                method = np.load(data_path_detail)
                coverage = method['coverage_db'][i]
                
                result_coverage[index_alpha] = coverage[-1]
                result_coverage_interval[:, index_alpha] = abs(method['coverage_db_interval'][i] - coverage[-1])

            temp = 10000
            ax2.errorbar(x_ticks + ((width + width / temp) * (index_method - 1.5)), result_coverage, yerr=result_coverage_interval, label=eval('grp.' + str(item))['fig_name'], capsize=24, fmt=eval('grp.' + str(item))['marker'], ecolor=eval('grp.' + str(item))['color'], elinewidth=6, markersize=12, color='black')
            # ax2.bar(x_ticks + ((width + width / temp) * (index_method - 1.5)), result_coverage, label=eval('grp.' + str(item))['fig_name'], color=eval('grp.' + str(item))['color'], width=width, align='center')
            
        ax2.set_ylabel(grp.title_db[i], fontsize=grp.font_size)
        ax2.set_xlabel(r'Prespecified coverage rate $\alpha$', fontsize=grp.font_size)

        ax2.set_xticks(range_alpha)
        ax2.grid(axis='y')
        ax2.legend(fontsize=30)
        plt.tick_params(labelsize=grp.ticks)
        save_path = data_path.replace('text', 'graph')
        mkdir(save_path, exist_ok=True)
        save_name = save_path + '/' + str(range_alpha) + grp.title_coverage_db[i]
        plt.savefig(save_name, bbox_inches='tight')

        plt.clf()
        plt.close()

        image_name = save_name.replace('pdf', 'png')
        cv(save_name, image_name)
