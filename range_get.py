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
  
def range_get(input_test, func_est, savepath):
    # only : dim = 1
    # if you want to create another figure, please change trial number.
    
    # data install 
            
    # get the pat
    fig_size = np.array([12, 8])
    fig = plt.figure(figsize=fig_size)

    input_test = input_test.reshape(-1)
    input_test_ord = np.argsort(input_test)
    input_test = np.sort(input_test).reshape(-1)
    
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['text.usetex'] = True
    
    ax = fig.add_subplot(1, 1, 1)

    a = func_est[0].flatten()
    a = a[input_test_ord]
    b = func_est[1].flatten()
    b = b[input_test_ord]
    #print(func_est[1])
    #method_func = func_est
    
    range_est = b - a
    
    # ax[index].scatter(input_test, observation_test, s=grp.dot_size, label='Observation', color='green')
    # ax[index].plot(input_test, output_test_true, label=r'True Function $\phi$', color='black', linewidth=grp.linewidth, linestyle='dashed')
    ax.plot(input_test, range_est, label='1', color='black', linewidth=grp.linewidth, alpha=0.95)
    # ax[index].plot(input_test, ground_truth_func[0], label='Ground truth : lower quantile', color='blue', linewidth=grp.linewidth, linestyle='dashed')
    # ax[index].plot(input_test, ground_truth_func[1], label='Ground truth : higher quantile', color='blue', linewidth=grp.linewidth, linestyle='dashed')

            
    # ax[index].set_title(str(eval('grp.' + str(item))['fig_name']), fontsize=grp.font_size)
    ax.set_xlabel('x', fontsize=grp.font_size)
    ax.set_ylabel('Range', fontsize=grp.font_size)
    #ax.set_ylim(0, 5)
    
    # plt.tick_params(labelsize=grp.ticks)
    ax.legend(fontsize=16)
    ax.grid()
    
    mkdir(savepath, exist_ok=True)
    save_name = savepath + '/fig_range.pdf'
    plt.savefig(save_name, bbox_inches='tight')

    plt.clf()
    plt.close()

    image_name = save_name.replace('pdf', 'png')
    cv(save_name, image_name)