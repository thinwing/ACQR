import numpy as np
import matplotlib.pyplot as plt
from os import makedirs as mkdir
from graph_package.tool_box.pdf2png import convert as cv
import sys
sys.path.append('../')
from integrate import get_path
from configuration import graph_config as grp
from configuration import config



# maybe it is useless
def comp_proposed_range_ave(data_path, list, loss=False, gamma=False):
    
    list_flatten = list.flatten()
    iteration = np.arange(config.Iter)
    range_alpha = config.range_alpha
    
    
    for index_alpha, range_alpha_temp in enumerate(range_alpha):
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(1, 1, 1)
        data_path_alpha = data_path + '/alpha=' + str(range_alpha_temp)
        for _, item in enumerate(list_flatten):

            data_path_detail, _ = get_path(data_path=data_path_alpha, method=item, loss=loss, gamma=gamma)
                        
            method = np.load(data_path_detail)
            range_ave = method['range_ave']
            
            ax.plot(iteration, range_ave, label=eval('grp.' + str(item))['fig_name'], color=eval('grp.' + str(item))['color'], linewidth=grp.linewidth)
        
        ax.set_xlabel('Iteration', fontsize=grp.font_size)
        ax.grid()
        ax.legend(fontsize=grp.font_size)
        ax.set_ylabel('Range', fontsize=grp.font_size)    

        save_path_alpha = data_path_alpha.replace('text', 'graph')
        if loss != False:
            save_path_alpha = save_path_alpha.replace('/alpha=', '/' + str(loss) + '/alpha=')
        mkdir(save_path_alpha, exist_ok=True)
        save_name = save_path_alpha + '/range.pdf'
        plt.savefig(save_name, bbox_inches='tight')        
        plt.clf()
        plt.close()
        image_name = save_name.replace('pdf', 'png')
        cv(save_name, image_name)
    
def comp_proposed_coverage_db(data_path, list, loss=False, gamma=False):
    list_flatten = list.flatten()
    iteration = np.arange(config.Iter)
    range_alpha = config.range_alpha
    
    
    for i in range(3):
        for index_alpha, range_alpha_temp in enumerate(range_alpha):
        
            fig = plt.figure(figsize=(8, 8))
            ax = fig.add_subplot(1, 1, 1)
            data_path_alpha = data_path + '/alpha=' + str(range_alpha_temp)
            for _, item in enumerate(list_flatten):

                data_path_detail, _ = get_path(data_path=data_path_alpha, method=item, loss=loss, gamma=gamma)

                
                method = np.load(data_path_detail)
                coverage_temp = method['coverage_db']
                coverage_db = coverage_temp[i]
                
                ax.plot(iteration, coverage_db, label=eval('grp.' + str(item))['fig_name'], color=eval('grp.' + str(item))['color'], linewidth=grp.linewidth)

            ax.set_xlabel('Iteration', fontsize=grp.font_size)
            ax.set_ylabel('Range error', fontsize=grp.font_size)
            plt.tick_params(labelsize=grp.ticks)
            ax.grid()
            ax.legend(fontsize=grp.font_size)

            save_path_alpha = data_path_alpha.replace('text', 'graph')
            if loss != False:
                save_path_alpha = save_path_alpha.replace('/alpha=', '/' + str(loss) + '/alpha=')
            mkdir(save_path_alpha, exist_ok=True)
            save_name = save_path_alpha + '/' + str(grp.title_coverage_db[i])
            plt.savefig(save_name, bbox_inches='tight')        
            plt.clf()
            plt.close()

            image_name = save_name.replace('pdf', 'png')
            cv(save_name, image_name)
        
    