import numpy as np
import matplotlib.pyplot as plt
from os import makedirs as mkdir
from graph_package.tool_box.pdf2png import convert as cv
import sys
import matplotlib.font_manager as font_manager
sys.path.append('../')
from configuration import graph_config as grp
from configuration import config
from integrate import get_path
from integrate import get_path_CQR


def gamma_coverage(data_path, loss_list, method='single_kernel', alpha=config.alpha_range):
    gamma = config.gamma

    # gamma_base = np.round((np.arange(11)) * 0.1, 2)
    gamma_base = gamma

    # tem = int(np.amin(result_coverage_db))
    
    #fig = plt.figure(figsize=(12, 8))

    #fig, ax1 = plt.subplots(figsize=(12, 8))
    fig = plt.figure(figsize=(12, 8))
    ax1 = fig.add_subplot(1, 1, 1)
    font_paths = [f.fname for f in font_manager.fontManager.ttflist]

    plt.tick_params(labelsize=grp.ticks)
    plt.rcParams['font.family'] = 'TimesNewRoman'    
    plt.rcParams['font.weight'] = 'normal'
    plt.rcParams['text.usetex'] = True

    import matplotlib.font_manager

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
                
                print(data_path_detail)
                method_result = np.load(data_path_detail)
                coverage = (method_result['coverage'][1] - method_result['coverage'][0]).reshape(-1)
                
                result_coverage[index_gamma] = coverage[-1]
        
            ax1.plot(gamma, result_coverage, label='conventional', color=eval('grp.' + str(item))['color'][0], linewidth=grp.linewidth, marker=eval('grp.' + str(item))['marker'], markersize=grp.marker_size)
#ここから追加パッチ
# proposed OCQKR
# comparative ACI                
        for index_gamma, gamma_temp in enumerate(gamma):
            # data load
            data_path_detail = 'result/text/dim=1/linear_expansion/sparse/outlier_rate=0.04/Iter=1000/alpha=0.95/online/pinball_moreau/' + '/\u03b3=' + str(gamma_temp) + '/CQR/single_kernel.npz'
            #'data_path_detail, _ = get_path_CQR(data_path=data_path_alpha, method=method, loss=item, gamma=gamma_temp)'
            
            print(data_path_detail)
            method_result = np.load(data_path_detail)
            coverage = (method_result['coverage'][1] - method_result['coverage'][0]).reshape(-1)
            
            result_coverage[index_gamma] = coverage[-1]
            print(result_coverage[index_gamma])
        ax1.plot(gamma, result_coverage, label='proposed', color='blue', linewidth=grp.linewidth, marker='o', markersize=grp.marker_size)
        for index_gamma, gamma_temp in enumerate(gamma):
            # data load
            data_path_detail = 'result/text/dim=1/linear_expansion/sparse/=0.04/Iter=3000/alpha=0.95/trial=10/online/pinball_moreau' + '/\u03b3=0.5' + '/ACI.npz'
            
            print(data_path_detail)
            method_result = np.load(data_path_detail)
            coverage = (method_result['coverage']).reshape(-1)
            
            result_coverage[index_gamma] = coverage[-1]
            print(result_coverage[index_gamma])
        ax1.plot(gamma, result_coverage, label='comparative', color='green', linewidth=grp.linewidth, marker='s', markersize=grp.marker_size)
    #ここまで追加パッチ

    #ax1.set_title('Range of r = ' + str(alpha) + ' and coverage by \u03b3', fontsize=grp.font_size)
    ax1.set_ylabel('Actual coverage rate', fontsize=grp.font_size)
    ax1.set_xlabel(r'$\gamma$', fontsize=grp.font_size)
    ax1.set_xscale('log')
    #ax1.set_ylim(0, 1)
    #ax1.set_xticks(gamma_base)
    ax1.grid()
    ax1.legend(fontsize=grp.font_size)
    ax1.hlines(0.95, 0, 1000, linewidth=4, color='black',linestyle='dashed')
    
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
        plt.rcParams['font.family'] = 'TimesNewRoman'
        plt.rcParams['font.weight'] = 'normal'    
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
                
                ax1.plot(gamma, result_coverage, label='conventional', color=eval('grp.' + str(item))['color'][0], linewidth=grp.linewidth, marker=eval('grp.' + str(item))['marker'], markersize=grp.marker_size)
#ここから追加パッチ
# proposed OCQKR
# comparative ACI
            for index_gamma, gamma_temp in enumerate(gamma):
                # data load
                data_path_detail = 'result/text/dim=1/linear_expansion/sparse/outlier_rate=0.04/Iter=1000/alpha=0.95/online/pinball_moreau/' + '/\u03b3=' + str(gamma_temp) + '/CQR/single_kernel.npz'
                
                method_result = np.load(data_path_detail)
                coverage = method_result['coverage_db'][i] 
                result_coverage[index_gamma] = coverage[-1]

            ax1.plot(gamma, result_coverage, label='proposed', color='blue', linewidth=grp.linewidth, marker='o', markersize=grp.marker_size)
            
            for index_gamma, gamma_temp in enumerate(gamma):
                # data load
                data_path_detail = 'result/text/dim=1/linear_expansion/sparse/=0.04/Iter=3000/alpha=0.95/trial=10/online/pinball_moreau' + '/\u03b3=0.5' + '/ACI.npz'
                
                method_result = np.load(data_path_detail)
                coverage = method_result['coverage_db'][i] 
                result_coverage[index_gamma] = coverage[-1]
            ax1.plot(gamma, result_coverage, label='comparative', color='green', linewidth=grp.linewidth, marker='s', markersize=grp.marker_size)
        #ここまで追加パッチ
        
        ax1.set_title('Error between ground truth and estimate result', fontsize=grp.font_size)
        ax1.set_ylabel('Error', fontsize=grp.font_size)
        ax1.set_xlabel(r'$\gamma$', fontsize=grp.font_size)
        ax1.set_xscale('log')
        #ax1.set_xticks(gamma_base)
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