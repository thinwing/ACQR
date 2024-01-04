import numpy as np
# import matplotlib
# del matplotlib.font_manager.weight_dict['roman']
# matplotlib.font_manager._rebuild()
from graph_package.tool_box.pdf2png import convert as cv
import matplotlib.pyplot as plt
from os import makedirs as mkdir
from configuration import config
import sys
sys.path.append('../')
from integrate import get_path
from configuration import graph_config as grp

def figACI(data_path, trial , list=np.array([['same_range'],['ground_truth']]),  loss=False, gamma=False):
    # only : dim = 1
    # if you want to create another figure, please change trial number.
    
    # data install
    
    data_path_temp = data_path + '/base/exp_data.npz'
    data = np.load(data_path_temp)
    
    #output_test_true = data['output_true_test']
    observation_test = data['observation_test']
    input_test = data['input_test'].reshape(-1)
    
    input_test_ord = np.argsort(input_test)
    
    input_test = np.sort(input_test).reshape(-1)
    #output_test_true = output_test_true[input_test_ord].reshape(-1)
    observation_test = observation_test[input_test_ord].reshape(-1)
        
    fig_size = np.array([12, 8])
    
    list_flatten = list.flatten()
    # ground_truth_path, _ = get_path(data_path=data_path, method='ground_truth')
    
    # ground_truth = np.load(ground_truth_path)
    # ground_truth_func = (ground_truth['func_est'])[:, input_test_ord]
    save_path = 'result/graph/dim=1/linear_expansion/sparse/outlier_rate=0.05/Iter=1000/alpha=0.95/trial=' + str(trial+1) +'/ACI'
    mkdir(save_path, exist_ok=True)
    
    for index, item in enumerate(list_flatten):
        fig = plt.figure(figsize=fig_size)
    
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams['text.usetex'] = True
    
        ax = fig.add_subplot(1, 1, 1)
        print(item)
    
        # get the path
        data_path_detail = 'result/text/dim=1/linear_expansion/sparse/=0.05/Iter=3000/alpha=0.95/trial=' + str(trial + 1) + '/online/pinball_moreau' + '/\u03b3=0.5/ACI.npz'
        
        # load the data
        method = np.load(data_path_detail)
        method_func = (method['func_est'])[:, input_test_ord]
        
        ax.scatter(input_test, observation_test, s=grp.dot_size, label='observation', color='green')
        #ax.plot(input_test, output_test_true, label=r'true function $\psi$', color='black', linewidth=grp.linewidth, linestyle='dashed')
        ax.fill_between(input_test, method_func[0].reshape(-1), method_func[1].reshape(-1), label=str(eval('grp.' + str(item))['fig_name']), facecolor='red', alpha=0.4)
        # ax[index].plot(input_test, ground_truth_func[0], label='Ground truth : lower quantile', color='blue', linewidth=grp.linewidth, linestyle='dashed')
        # ax[index].plot(input_test, ground_truth_func[1], label='Ground truth : higher quantile', color='blue', linewidth=grp.linewidth, linestyle='dashed')
                
        # ax[index].set_title(str(eval('grp.' + str(item))['fig_name']), fontsize=grp.font_size)
        ax.set_xlabel('$x$', fontsize=grp.font_size)
        ax.set_ylabel('$y$', fontsize=grp.font_size)
        ax.set_ylim(-5, 10)
        #ax.set_xlim(0, 1)
        
        ax.legend(fontsize=32)
        plt.tick_params(labelsize=32)
        ax.grid()
        save_name = save_path + '/fig_CQR' + str(item)+ '.pdf'
        plt.savefig(save_name, bbox_inches='tight')

        plt.clf()
        plt.close()

        image_name = save_name.replace('pdf', 'png')
        cv(save_name, image_name)
    
def fig_rangeACI(data_path, trial , list=np.array([['same_range'],['ground_truth']]), loss=False, gamma=False):
    # only : dim = 1
    # if you want to create another figure, please change trial number.
    
    # data install 
    print(data_path)
    data_path_temp = data_path + '/base/exp_data.npz'
    data = np.load(data_path_temp)
    
    # output_test_true = data['output_true_test']
    # observation_test = data['observation_test']
    input_test = data['input_test'].reshape(-1)
    
    inputa,inputb,input_test = np.split(input_test,3)
    input_test_ord = np.argsort(input_test)
    
    input_test = np.sort(input_test).reshape(-1)
    # output_test_true = output_test_true[input_test_ord].reshape(-1)
    #observation_test = observation_test[input_test_ord].reshape(-1)
        
    fig_size = np.array([12, 8])
    fig = plt.figure(figsize=fig_size)
    
    
    list_flatten = list.flatten()
    width = int(round(len(list_flatten) / 2, 1))
    
    ground_truth_path, _ = get_path(data_path=data_path, method='ground_truth')
    
    ground_truth = np.load(ground_truth_path)
    ground_truth_func = (ground_truth['func_est'])[:, input_test_ord]
    range_true = ground_truth_func[1] - ground_truth_func[0]
        
    ax = fig.add_subplot(1, 1, 1)
    
    ax.plot(input_test, range_true, label='True range', color='black')
    
    for index, item in enumerate(list_flatten):
    
        # get the path
        data_path_detail, _= get_path(data_path=data_path, method=item, loss=loss, gamma=gamma)
        
        # load the data
        method = np.load(data_path_detail)
        method_func = (method['func_est'])[:, input_test_ord]
        print('rinchan')
        print(method_func[1])
        print(method_func[0])
        
        range_est = method_func[1] - method_func[0]
        print(range_est)
        
        # ax[index].scatter(input_test, observation_test, s=grp.dot_size, label='Observation', color='green')
        # ax[index].plot(input_test, output_test_true, label=r'True Function $\phi$', color='black', linewidth=grp.linewidth, linestyle='dashed')
        ax.plot(input_test, range_est, label=str(eval('grp.' + str(item))['fig_name']), color=eval('grp.' + str(item))['color'], linewidth=grp.linewidth, alpha=eval('grp.' + str(item))['alpha'])
        # ax[index].plot(input_test, ground_truth_func[0], label='Ground truth : lower quantile', color='blue', linewidth=grp.linewidth, linestyle='dashed')
        # ax[index].plot(input_test, ground_truth_func[1], label='Ground truth : higher quantile', color='blue', linewidth=grp.linewidth, linestyle='dashed')

                
        # ax[index].set_title(str(eval('grp.' + str(item))['fig_name']), fontsize=grp.font_size)
        ax.set_xlabel('x', fontsize=grp.font_size)
        ax.set_ylabel('Range', fontsize=grp.font_size)
        ax.set_ylim(-5, 10)
        
        # plt.tick_params(labelsize=grp.ticks)
        ax.legend(fontsize=16)
        ax.grid()
    
    save_path = 'result/graph/dim=1/linear_expansion/sparse/=0.05/Iter=3000/alpha=0.95/trial=' + str(trial+1) +'/ACI'
    mkdir(save_path, exist_ok=True)
    save_name = save_path + '/fig_range.pdf'
    plt.savefig(save_name, bbox_inches='tight')

    plt.clf()
    plt.close()

    image_name = save_name.replace('pdf', 'png')
    cv(save_name, image_name)

def figCQR(data_path, trial , list=np.array([['same_range'],['ground_truth']]),  loss=False, gamma=False):
    # only : dim = 1
    # if you want to create another figure, please change trial number.
    
    # data install
    
    data_path_tem = data_path + '/base/exp_data.npz'
    data_path_temp = data_path_tem.replace('/trial', '/lo/trial')
    data = np.load(data_path_temp)
    
    input_path = 'result/text/dim=1/linear_expansion/sparse/outlier_rate=0.05/Iter=1000/alpha=0.95/lo/trial=1/online/pinball_moreau' + '/\u03b3=' + str(config.gamma_default) + '/CQR/single_kernel.npz'
    path = np.load(input_path)
    path_test = data['input_test']
    input_te = path_test[2000:2500]
    input_testc = input_te.reshape(-1)

    input_test = path['input_te'].reshape(-1)
    print(len(input_test))

    #output_test_true = data['output_true_test']
    observation = data['observation_test']
    observation_test = observation[2000:2500]
    input_test_ord = np.argsort(input_test)
    input_test_ordc = np.argsort(input_testc)
    
    input_testc = np.sort(input_testc).reshape(-1)
    input_test = np.sort(input_test).reshape(-1)
    #output_test_true = output_test_true[input_test_ord].reshape(-1)
    observation_test = observation_test[input_test_ordc].reshape(-1)
        
    fig_size = np.array([12, 8])
    
    list_flatten = list.flatten()
    print('flat')
    print(list_flatten)
    
    # ground_truth_path, _ = get_path(data_path=data_path, method='ground_truth')
    
    # ground_truth = np.load(ground_truth_path)
    # ground_truth_func = (ground_truth['func_est'])[:, input_test_ord]
    save_path = 'result/graph/dim=1/linear_expansion/sparse/outlier_rate=0.05/Iter=1000/alpha=0.95/trial=' + str(trial+1) +'/CQR/lo'
    mkdir(save_path, exist_ok=True)
    
    for index, item in enumerate(list_flatten):
        fig = plt.figure(figsize=fig_size)
    
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams['text.usetex'] = True
    
        ax = fig.add_subplot(1, 1, 1)
        print(item)
    
        # get the path
        data_path_detail = 'result/text/dim=1/linear_expansion/sparse/outlier_rate=0.05/Iter=1000/alpha=0.95/lo/trial=' + str(trial + 1) + '/online/pinball_moreau' + '/\u03b3=' + str(config.gamma_default) + '/CQR/single_kernel.npz'

        # load the data
        method = np.load(data_path_detail)
        print('method')
        print(method['func_est'])
        method_func = (method['func_est'])[:, input_test_ordc]
        
        ax.scatter(input_testc, observation_test, s=grp.dot_size, label='observation', color='green')
        #ax.plot(input_test, output_test_true, label=r'true function $\psi$', color='black', linewidth=grp.linewidth, linestyle='dashed')
        ax.fill_between(input_test, method_func[0].reshape(-1), method_func[1].reshape(-1), label=str(eval('grp.' + str(item))['fig_name']), facecolor='red', alpha=0.4)
        # ax[index].plot(input_test, ground_truth_func[0], label='Ground truth : lower quantile', color='blue', linewidth=grp.linewidth, linestyle='dashed')
        # ax[index].plot(input_test, ground_truth_func[1], label='Ground truth : higher quantile', color='blue', linewidth=grp.linewidth, linestyle='dashed')
                
        # ax[index].set_title(str(eval('grp.' + str(item))['fig_name']), fontsize=grp.font_size)
        ax.set_xlabel('$x$', fontsize=grp.font_size)
        ax.set_ylabel('$y$', fontsize=grp.font_size)
        
        #調整しよう
        ax.set_ylim(-25, 25)
        ax.set_xlim(0, 1)
        
        ax.legend(fontsize=32)
        plt.tick_params(labelsize=32)
        ax.grid()
        save_name = save_path + '/fig_' + str(item)+ '.pdf'
        plt.savefig(save_name, bbox_inches='tight')

        plt.clf()
        plt.close()

        image_name = save_name.replace('pdf', 'png')
        cv(save_name, image_name)
    
def fig_rangeCQR(data_path, trial , list=np.array([['same_range'],['ground_truth']]), loss=False, gamma=False):
    # only : dim = 1
    # if you want to create another figure, please change trial number.
    
    # data install 
    print(data_path)
    data_path_temp = data_path + '/base/exp_data.npz'
    data = np.load(data_path_temp)
    
    # output_test_true = data['output_true_test']
    observation_test = data['observation_test']
    input_test = data['input_test'].reshape(-1)
    
    input_test_ord = np.argsort(input_test)
    
    input_test = np.sort(input_test).reshape(-1)
    # output_test_true = output_test_true[input_test_ord].reshape(-1)
    observation_test = observation_test[input_test_ord].reshape(-1)
        
    fig_size = np.array([12, 8])
    fig = plt.figure(figsize=fig_size)
    
    
    list_flatten = list.flatten()
    width = int(round(len(list_flatten) / 2, 1))
    
    ground_truth_path, _ = get_path(data_path=data_path, method='ground_truth')
    
    ground_truth = np.load(ground_truth_path)
    ground_truth_func = (ground_truth['func_est'])[:, input_test_ord]
    range_true = ground_truth_func[1] - ground_truth_func[0]
        
    ax = fig.add_subplot(1, 1, 1)
    
    ax.plot(input_test, range_true, label='True range', color='black')
    
    for index, item in enumerate(list_flatten):
    
        # get the path
        data_path_detail, _= get_path(data_path=data_path, method=item, loss=loss, gamma=gamma)
        
        # load the data
        method = np.load(data_path_detail)
        method_func = (method['func_est'])[:, input_test_ord]
        
        range_est = method_func[1] - method_func[0]
        
        # ax[index].scatter(input_test, observation_test, s=grp.dot_size, label='Observation', color='green')
        # ax[index].plot(input_test, output_test_true, label=r'True Function $\phi$', color='black', linewidth=grp.linewidth, linestyle='dashed')
        ax.plot(input_test, range_est, label=str(eval('grp.' + str(item))['fig_name']), color=eval('grp.' + str(item))['color'], linewidth=grp.linewidth, alpha=eval('grp.' + str(item))['alpha'])
        # ax[index].plot(input_test, ground_truth_func[0], label='Ground truth : lower quantile', color='blue', linewidth=grp.linewidth, linestyle='dashed')
        # ax[index].plot(input_test, ground_truth_func[1], label='Ground truth : higher quantile', color='blue', linewidth=grp.linewidth, linestyle='dashed')

                
        # ax[index].set_title(str(eval('grp.' + str(item))['fig_name']), fontsize=grp.font_size)
        ax.set_xlabel('x', fontsize=grp.font_size)
        ax.set_ylabel('Range', fontsize=grp.font_size)
        ax.set_ylim(-5, 15)
        
        # plt.tick_params(labelsize=grp.ticks)
        ax.legend(fontsize=16)
        ax.grid()
    
    save_path = 'result/graph/dim=1/linear_expansion/sparse/outlier_rate=0.05/Iter=1000/alpha=0.95/trial=' + str(trial+1) +'/CQR'
    mkdir(save_path, exist_ok=True)
    save_name = save_path + '/fig_range_CQR.pdf'
    plt.savefig(save_name, bbox_inches='tight')

    plt.clf()
    plt.close()

    image_name = save_name.replace('pdf', 'png')
    cv(save_name, image_name)