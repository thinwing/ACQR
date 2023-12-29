import numpy as np
import matplotlib.pyplot as plt
from os import makedirs as mkdir
from graph_package.tool_box.pdf2png import convert as cv
import sys
sys.path.append('../')
from integrate import get_path
from configuration import graph_config as grp
from configuration import config



def comp_proposed_coverage_both(data_path, list, loss=False, gamma=False):
    
    iteration = np.arange(config.Iter)
    list_flatten = list.flatten()
    alpha_all = config.alpha_all
    range_alpha = config.range_alpha
    fig_size = (12.0 * len(list_flatten), 8.0)

    for index_alpha, range_alpha_temp in enumerate(alpha_all):
        fig = plt.figure(figsize=fig_size) 
        plt.tick_params(labelsize=grp.ticks)
        plt.rcParams['font.family'] = 'Times New Roman'    
        plt.rcParams['text.usetex'] = True
        # fig = plt.figure(figsize=(24.0, 16.0))
        ax = []
        data_path_alpha = data_path + '/alpha=' + str(range_alpha[index_alpha])
        
        for index, item in enumerate(list_flatten):
            ax.append(fig.add_subplot(1, len(list_flatten), (index + 1)))
            

            data_path_detail, _ = get_path(data_path=data_path_alpha, method=item, loss=loss, gamma=gamma)
                        
            method = np.load(data_path_detail)
            coverage = method['coverage']

            ax[index].plot(iteration, coverage[1], label=eval('grp.' + str(item))['fig_name'] + r' : $\alpha$ = ' + str(range_alpha_temp[1]), color='red', linewidth=grp.linewidth)
            ax[index].plot(iteration, range_alpha_temp[1] * np.ones([config.Iter, 1]), label=str(range_alpha_temp[1]), color='green', linewidth=grp.linewidth, linestyle='dashed')

            ax[index].plot(iteration, coverage[0], label=eval('grp.' + str(item))['fig_name'] + r' : $\alpha$ = ' + str(range_alpha_temp[0]), color='orange', linewidth=grp.linewidth)
            ax[index].plot(iteration, range_alpha_temp[0] * np.ones([config.Iter, 1]), label=str(range_alpha_temp[0]), color='green', linewidth=grp.linewidth, linestyle='dashed')


            ax[index].set_xlabel('Iteration',fontsize=grp.font_size)
            ax[index].set_ylabel('Quantile')
            #ax[index].set_ylim(0, 1)

            ax[index].grid()
            ax[index].legend(fontsize=grp.font_size)

        save_path_alpha = data_path_alpha.replace('text','graph') 
        if loss != False:
            save_path_alpha = save_path_alpha.replace('/alpha=', '/' + str(loss) + '/alpha=')
        mkdir(save_path_alpha, exist_ok=True)
        save_name = save_path_alpha + '/ratio.pdf'
        plt.savefig(save_name, bbox_inches='tight')

        plt.clf()
        plt.close()
        image_name = save_name.replace('pdf', 'png')
        cv(save_name, image_name)


        
def comp_proposed_coverage(data_path, list, loss=False, gamma=False):
    list_flatten = list.flatten()
    iteration = np.arange(config.Iter)
    range_alpha = config.range_alpha
    print('range_alpha')
    print(range_alpha)
    
    range_alpha_temp = 0.95
    ground_alpha = np.ones(config.Iter) * range_alpha_temp
    
    #fig = plt.figure(figsize=(12, 4))
    fig, ax = plt.subplots(figsize=(12, 4))
    plt.tick_params(labelsize=grp.ticks)
    plt.rcParams['font.family'] = 'Times New Roman'    
    plt.rcParams['text.usetex'] = True
    #ax = fig.add_subplot(1, 1, 1)
    ax.plot(iteration, ground_alpha, label='ground truth', color='black', linewidth=grp.linewidth, linestyle='dashdot')
    data_path_alpha = data_path + '/alpha=' + str(range_alpha_temp)
    
    for _, item in enumerate(list_flatten):
        print('data_path_alpha')
        print(data_path_alpha)
        data_path_detail, _ = get_path(data_path=data_path_alpha, method=item, loss=loss, gamma=gamma)
                    
        method = np.load(data_path_detail)
        coverage_temp = method['coverage']
        coverage = coverage_temp[1] - coverage_temp[0]
        
        ax.plot(iteration, coverage, label=eval('grp.' + str(item))['fig_name'], color=eval('grp.' + str(item))['color'], linewidth=grp.linewidth)
    
    ax.set_ylabel('Coverage', fontsize=grp.font_size)
    ax.set_xlabel('Iteration', fontsize=grp.font_size)
    ax.grid()
    ax.legend(fontsize=grp.font_size)
    #ax.set_ylim(0, 1)

    save_path_alpha = data_path_alpha.replace('text','graph')
    if loss != False:
        save_path_alpha = save_path_alpha.replace('/alpha=', '/' + str(loss) + '/alpha=')
    mkdir(save_path_alpha, exist_ok=True)

    save_name = save_path_alpha + '/coverage.pdf'        
    plt.savefig(save_name, bbox_inches='tight')        
    plt.clf()
    plt.close()

    image_name = save_name.replace('pdf', 'png')
    cv(save_name, image_name)
