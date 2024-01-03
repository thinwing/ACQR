from configuration import config
from configuration import address
from configuration import graph_config as grp
import numpy as np
from os import makedirs as mkdir

import scipy as sp
import scipy.stats as st

def get_path(data_path, method, loss=False, gamma=False):
    if eval('grp.' + str(method))['loss'] == 'None':
        data_path_add = '/' + str(eval('grp.' + str(method))['address']) 
        data_path_detail = data_path + data_path_add + '/' + str(eval('grp.' + str(method))['save_name']) + '.npz'
        
    else: 
        if loss == False:    
            loss_dict = eval('grp.' + eval('grp.' + str(method))['loss'])
            gamma = loss_dict['gamma']
        else:
            loss_dict = eval('grp.' + str(loss))
        data_path_add = '/' + str(eval('grp.' + str(method))['address']) + '/'  + str(loss_dict['loss']) + '/\u03b3=' + str(gamma)          
        data_path_detail = data_path + data_path_add + '/' + str(eval('grp.' + str(method))['save_name']) + '.npz'
    return data_path_detail, data_path_add

def get_path_CQR(data_path, method, loss=False, gamma=False):
    if eval('grp.' + str(method))['loss'] == 'None':
        data_path_add = '/' + str(eval('grp.' + str(method))['address']) + '/CQR'
        data_path_detail = data_path + data_path_add + '/' + str(eval('grp.' + str(method))['save_name']) + '.npz'
        
    else: 
        if loss == False:    
            loss_dict = eval('grp.' + eval('grp.' + str(method))['loss'])
            gamma = loss_dict['gamma']
        else:
            loss_dict = eval('grp.' + str(loss))
        data_path_add = '/' + str(eval('grp.' + str(method))['address']) + '/'  + str(loss_dict['loss']) + '/\u03b3=' + str(gamma)  + '/CQR'         
        data_path_detail = data_path + data_path_add + '/' + str(eval('grp.' + str(method))['save_name']) + '.npz'
    return data_path_detail, data_path_add

def data_integrate(data_path, method, loss, gamma, trial=config.trial):
    cov_temp = np.zeros(trial)
    cov_db_temp = np.zeros([trial, 3])
    cov_db_interval = np.zeros([3, 2])

    alpha_rel = 0.95
    deg_free = trial - 1

    for i in range(trial):
        # get path
        data_path_temp = data_path + '/' + 'trial=' + str(i + 1)
        data_path_detail, _ = get_path(data_path=data_path_temp, method=method, loss=loss, gamma=gamma)
        
        result = np.load(data_path_detail)
        
        if i == 0:
            coverage = result['coverage'] / trial
            coverage_all = result['coverage_all'] / trial
            coverage_db = result['coverage_db'] / trial
            range_ave = result['range_ave'] / trial

        else:
            coverage += result['coverage'] / trial        
            coverage_all += result['coverage_all'] / trial
            coverage_db += result['coverage_db'] / trial
            range_ave += result['range_ave'] / trial
        
        cov_db_temp[i] = result['coverage_db'][:, -1].reshape(-1) 
        cov_temp[i] = (((result['coverage'])[1] - (result['coverage'])[0]).reshape(-1))[-1]

    cov_ave = st.tmean(cov_temp)
    cov_scale = np.sqrt(st.tvar(cov_temp) / trial)

    for j in range(3):
        cov_ave_db = st.tmean(cov_db_temp[:, j])
        cov_scale_db = np.sqrt(st.tvar(cov_db_temp[:, j]) / trial)
        cov_db_interval[j] = st.t.interval(alpha_rel, deg_free, loc=cov_ave_db, scale=cov_scale_db)
    
    cov_interval = st.t.interval(alpha_rel, deg_free, loc=cov_ave, scale=cov_scale)
    # save
    save_path, save_path_add = get_path(data_path=data_path, method=method, loss=loss, gamma=gamma)
    print(save_path)
    mkdir(data_path+save_path_add, exist_ok=True)
    np.savez_compressed(save_path, coverage=coverage, coverage_all=coverage_all, coverage_interval=cov_interval, coverage_db=10 * np.log10(coverage_db), coverage_db_interval=(10 * np.log10(cov_db_interval)), range_ave=range_ave)

def data_integrate_hi(data_path, method, loss, gamma, trial=config.trial):
    cov_temp = np.zeros(trial)
    cov_db_temp = np.zeros([trial, 3])
    cov_db_interval = np.zeros([3, 2])

    alpha_rel = 0.95
    deg_free = trial - 1

    for i in range(trial):
        # get path
        data_path_temp = data_path + '/hi/' + 'trial=' + str(i + 1)
        data_path_detail, _ = get_path(data_path=data_path_temp, method=method, loss=loss, gamma=gamma)
        result = np.load(data_path_detail)
        
        if i == 0:
            coverage = result['coverage'] / trial
            coverage_all = result['coverage_all'] / trial
            coverage_db = result['coverage_db'] / trial
            range_ave = result['range_ave'] / trial

        else:
            coverage += result['coverage'] / trial        
            coverage_all += result['coverage_all'] / trial
            coverage_db += result['coverage_db'] / trial
            range_ave += result['range_ave'] / trial
        
        cov_db_temp[i] = result['coverage_db'][:, -1].reshape(-1) 
        cov_temp[i] = (((result['coverage'])[1] - (result['coverage'])[0]).reshape(-1))[-1]

    cov_ave = st.tmean(cov_temp)
    cov_scale = np.sqrt(st.tvar(cov_temp) / trial)

    for j in range(3):
        cov_ave_db = st.tmean(cov_db_temp[:, j])
        cov_scale_db = np.sqrt(st.tvar(cov_db_temp[:, j]) / trial)
        cov_db_interval[j] = st.t.interval(alpha_rel, deg_free, loc=cov_ave_db, scale=cov_scale_db)
    
    cov_interval = st.t.interval(alpha_rel, deg_free, loc=cov_ave, scale=cov_scale)
    # save
    save_path, save_path_add = get_path(data_path=data_path, method=method, loss=loss, gamma=gamma)
    data_path = data_path.replace('alpha=0.95', 'alpha=0.95/hi')
    save_path = save_path.replace('alpha=0.95/', 'alpha=0.95/hi/')
    mkdir(data_path+save_path_add, exist_ok=True)
    np.savez_compressed(save_path, coverage=coverage, coverage_all=coverage_all, coverage_interval=cov_interval, coverage_db=10 * np.log10(coverage_db), coverage_db_interval=(10 * np.log10(cov_db_interval)), range_ave=range_ave)

def data_integrate_lo(data_path, method, loss, gamma, trial=config.trial):
    cov_temp = np.zeros(trial)
    cov_db_temp = np.zeros([trial, 3])
    cov_db_interval = np.zeros([3, 2])

    alpha_rel = 0.95
    deg_free = trial - 1

    for i in range(trial):
        # get path
        data_path_temp = data_path + '/lo/' + 'trial=' + str(i + 1)
        data_path_detail, _ = get_path(data_path=data_path_temp, method=method, loss=loss, gamma=gamma)
        
        result = np.load(data_path_detail)
        
        if i == 0:
            coverage = result['coverage'] / trial
            coverage_all = result['coverage_all'] / trial
            coverage_db = result['coverage_db'] / trial
            range_ave = result['range_ave'] / trial

        else:
            coverage += result['coverage'] / trial        
            coverage_all += result['coverage_all'] / trial
            coverage_db += result['coverage_db'] / trial
            range_ave += result['range_ave'] / trial
        
        cov_db_temp[i] = result['coverage_db'][:, -1].reshape(-1) 
        cov_temp[i] = (((result['coverage'])[1] - (result['coverage'])[0]).reshape(-1))[-1]

    cov_ave = st.tmean(cov_temp)
    cov_scale = np.sqrt(st.tvar(cov_temp) / trial)

    for j in range(3):
        cov_ave_db = st.tmean(cov_db_temp[:, j])
        cov_scale_db = np.sqrt(st.tvar(cov_db_temp[:, j]) / trial)
        cov_db_interval[j] = st.t.interval(alpha_rel, deg_free, loc=cov_ave_db, scale=cov_scale_db)
    
    cov_interval = st.t.interval(alpha_rel, deg_free, loc=cov_ave, scale=cov_scale)
    # save
    save_path, save_path_add = get_path(data_path=data_path, method=method, loss=loss, gamma=gamma)
    data_path = data_path.replace('alpha=0.95', 'alpha=0.95/lo')
    save_path = save_path.replace('alpha=0.95/', 'alpha=0.95/lo/')
    print(save_path)
    mkdir(data_path+save_path_add, exist_ok=True)
    np.savez_compressed(save_path, coverage=coverage, coverage_all=coverage_all, coverage_interval=cov_interval, coverage_db=10 * np.log10(coverage_db), coverage_db_interval=(10 * np.log10(cov_db_interval)), range_ave=range_ave)

def data_integrate_hal(data_path, method, loss, gamma, trial=config.trial):
    cov_temp = np.zeros(trial)
    cov_db_temp = np.zeros([trial, 3])
    cov_db_interval = np.zeros([3, 2])

    alpha_rel = 0.95
    deg_free = trial - 1

    for i in range(trial):
        # get path
        data_path_temp = data_path + '/hal/' + 'trial=' + str(i + 1)
        data_path_detail, _ = get_path(data_path=data_path_temp, method=method, loss=loss, gamma=gamma)
        
        result = np.load(data_path_detail)
        
        if i == 0:
            coverage = result['coverage'] / trial
            coverage_all = result['coverage_all'] / trial
            coverage_db = result['coverage_db'] / trial
            range_ave = result['range_ave'] / trial

        else:
            coverage += result['coverage'] / trial        
            coverage_all += result['coverage_all'] / trial
            coverage_db += result['coverage_db'] / trial
            range_ave += result['range_ave'] / trial
        
        cov_db_temp[i] = result['coverage_db'][:, -1].reshape(-1) 
        cov_temp[i] = (((result['coverage'])[1] - (result['coverage'])[0]).reshape(-1))[-1]

    cov_ave = st.tmean(cov_temp)
    cov_scale = np.sqrt(st.tvar(cov_temp) / trial)

    for j in range(3):
        cov_ave_db = st.tmean(cov_db_temp[:, j])
        cov_scale_db = np.sqrt(st.tvar(cov_db_temp[:, j]) / trial)
        cov_db_interval[j] = st.t.interval(alpha_rel, deg_free, loc=cov_ave_db, scale=cov_scale_db)
    
    cov_interval = st.t.interval(alpha_rel, deg_free, loc=cov_ave, scale=cov_scale)
    # save
    save_path, save_path_add = get_path(data_path=data_path, method=method, loss=loss, gamma=gamma)
    data_path = data_path.replace('alpha=0.95', 'alpha=0.95/hal')
    save_path = save_path.replace('alpha=0.95/', 'alpha=0.95/hal/')
    print(save_path)
    mkdir(data_path+save_path_add, exist_ok=True)
    np.savez_compressed(save_path, coverage=coverage, coverage_all=coverage_all, coverage_interval=cov_interval, coverage_db=10 * np.log10(coverage_db), coverage_db_interval=(10 * np.log10(cov_db_interval)), range_ave=range_ave)

def data_integrate_CQR(data_path, method, loss, gamma, trial=config.trial):
    cov_temp = np.zeros(trial)
    cov_db_temp = np.zeros([trial, 3])
    cov_db_interval = np.zeros([3, 2])
    #input_neo = np.zeros([trial,config.Iter_CQR])
    neo = int(config.Iter/6)
    input_neo = np.zeros([trial,neo])

    alpha_rel = 0.95
    deg_free = trial - 1

    for i in range(trial):
        # get path
        data_path_temp = data_path + '/' + 'trial=' + str(i + 1)
        data_path_detail, _ = get_path_CQR(data_path=data_path_temp, method=method, loss=loss, gamma=gamma)

        result = np.load(data_path_detail)
        
        if i == 0:
            coverage = result['coverage'] / trial
            coverage_all = result['coverage_all'] / trial
            coverage_db = result['coverage_db'] / trial
            range_ave = result['range_ave'] / trial
            input_neo[0,:] = result['input_te'].reshape(-1)

        else:
            coverage += result['coverage'] / trial        
            coverage_all += result['coverage_all'] / trial
            coverage_db += result['coverage_db'] / trial
            range_ave += result['range_ave'] / trial
            input_neo[i,:] = result['input_te'].reshape(-1)
        
        cov_db_temp[i] = result['coverage_db'][:, -1].reshape(-1) 
        cov_temp[i] = (((result['coverage'])[1] - (result['coverage'])[0]).reshape(-1))[-1]

    cov_ave = st.tmean(cov_temp)
    cov_scale = np.sqrt(st.tvar(cov_temp) / trial)

    for j in range(3):
        cov_ave_db = st.tmean(cov_db_temp[:, j])
        cov_scale_db = np.sqrt(st.tvar(cov_db_temp[:, j]) / trial)
        cov_db_interval[j] = st.t.interval(alpha_rel, deg_free, loc=cov_ave_db, scale=cov_scale_db)
    
    cov_interval = st.t.interval(alpha_rel, deg_free, loc=cov_ave, scale=cov_scale)
    # save
    save_path, save_path_add = get_path_CQR(data_path=data_path, method=method, loss=loss, gamma=gamma)
    print(save_path)
    mkdir(data_path+save_path_add, exist_ok=True)
    np.savez_compressed(save_path, coverage=coverage, coverage_all=coverage_all, coverage_interval=cov_interval, coverage_db=10 * np.log10(coverage_db), coverage_db_interval=(10 * np.log10(cov_db_interval)), range_ave=range_ave, input_te=input_neo)

def data_integrate_CQRhi(data_path, method, loss, gamma, trial=config.trial):
    cov_temp = np.zeros(trial)
    cov_db_temp = np.zeros([trial, 3])
    cov_db_interval = np.zeros([3, 2])
    #input_neo = np.zeros([trial,config.Iter_CQR])
    neo = int(config.Iter/6)
    input_neo = np.zeros([trial,neo])

    alpha_rel = 0.95
    deg_free = trial - 1

    for i in range(trial):
        # get path
        data_path_temp = data_path + '/hi/' + 'trial=' + str(i + 1)
        data_path_detail, _ = get_path_CQR(data_path=data_path_temp, method=method, loss=loss, gamma=gamma)

        result = np.load(data_path_detail)
        
        if i == 0:
            coverage = result['coverage'] / trial
            coverage_all = result['coverage_all'] / trial
            coverage_db = result['coverage_db'] / trial
            range_ave = result['range_ave'] / trial
            input_neo[0,:] = result['input_te'].reshape(-1)

        else:
            coverage += result['coverage'] / trial        
            coverage_all += result['coverage_all'] / trial
            coverage_db += result['coverage_db'] / trial
            range_ave += result['range_ave'] / trial
            input_neo[i,:] = result['input_te'].reshape(-1)
        
        cov_db_temp[i] = result['coverage_db'][:, -1].reshape(-1) 
        cov_temp[i] = (((result['coverage'])[1] - (result['coverage'])[0]).reshape(-1))[-1]

    cov_ave = st.tmean(cov_temp)
    cov_scale = np.sqrt(st.tvar(cov_temp) / trial)

    for j in range(3):
        cov_ave_db = st.tmean(cov_db_temp[:, j])
        cov_scale_db = np.sqrt(st.tvar(cov_db_temp[:, j]) / trial)
        cov_db_interval[j] = st.t.interval(alpha_rel, deg_free, loc=cov_ave_db, scale=cov_scale_db)
    
    cov_interval = st.t.interval(alpha_rel, deg_free, loc=cov_ave, scale=cov_scale)
    # save

    save_path, save_path_add = get_path_CQR(data_path=data_path, method=method, loss=loss, gamma=gamma)
    data_path = data_path.replace('alpha=0.95', 'alpha=0.95/hi')
    save_path = save_path.replace('alpha=0.95/', 'alpha=0.95/hi/')
    print(save_path)
    mkdir(data_path+save_path_add, exist_ok=True)
    np.savez_compressed(save_path, coverage=coverage, coverage_all=coverage_all, coverage_interval=cov_interval, coverage_db=10 * np.log10(coverage_db), coverage_db_interval=(10 * np.log10(cov_db_interval)), range_ave=range_ave, input_te=input_neo)

def data_integrate_CQRlo(data_path, method, loss, gamma, trial=config.trial):
    cov_temp = np.zeros(trial)
    cov_db_temp = np.zeros([trial, 3])
    cov_db_interval = np.zeros([3, 2])
    #input_neo = np.zeros([trial,config.Iter_CQR])
    neo = int(config.Iter/6)
    input_neo = np.zeros([trial,neo])

    alpha_rel = 0.95
    deg_free = trial - 1

    for i in range(trial):
        # get path
        data_path_temp = data_path + '/lo/' + 'trial=' + str(i + 1)
        data_path_detail, _ = get_path_CQR(data_path=data_path_temp, method=method, loss=loss, gamma=gamma)

        result = np.load(data_path_detail)
        
        if i == 0:
            coverage = result['coverage'] / trial
            coverage_all = result['coverage_all'] / trial
            coverage_db = result['coverage_db'] / trial
            range_ave = result['range_ave'] / trial
            input_neo[0,:] = result['input_te'].reshape(-1)

        else:
            coverage += result['coverage'] / trial        
            coverage_all += result['coverage_all'] / trial
            coverage_db += result['coverage_db'] / trial
            range_ave += result['range_ave'] / trial
            input_neo[i,:] = result['input_te'].reshape(-1)
        
        cov_db_temp[i] = result['coverage_db'][:, -1].reshape(-1) 
        cov_temp[i] = (((result['coverage'])[1] - (result['coverage'])[0]).reshape(-1))[-1]

    cov_ave = st.tmean(cov_temp)
    cov_scale = np.sqrt(st.tvar(cov_temp) / trial)

    for j in range(3):
        cov_ave_db = st.tmean(cov_db_temp[:, j])
        cov_scale_db = np.sqrt(st.tvar(cov_db_temp[:, j]) / trial)
        cov_db_interval[j] = st.t.interval(alpha_rel, deg_free, loc=cov_ave_db, scale=cov_scale_db)
    
    cov_interval = st.t.interval(alpha_rel, deg_free, loc=cov_ave, scale=cov_scale)
    # save
    save_path, save_path_add = get_path_CQR(data_path=data_path, method=method, loss=loss, gamma=gamma)
    data_path = data_path.replace('alpha=0.95', 'alpha=0.95/lo')
    save_path = save_path.replace('alpha=0.95/', 'alpha=0.95/lo/')
    print(save_path)
    mkdir(data_path+save_path_add, exist_ok=True)
    np.savez_compressed(save_path, coverage=coverage, coverage_all=coverage_all, coverage_interval=cov_interval, coverage_db=10 * np.log10(coverage_db), coverage_db_interval=(10 * np.log10(cov_db_interval)), range_ave=range_ave, input_te=input_neo)

def data_integrate_CQRhal(data_path, method, loss, gamma, trial=config.trial):
    cov_temp = np.zeros(trial)
    cov_db_temp = np.zeros([trial, 3])
    cov_db_interval = np.zeros([3, 2])
    #input_neo = np.zeros([trial,config.Iter_CQR])
    neo = int(config.Iter/6)
    input_neo = np.zeros([trial,neo])

    alpha_rel = 0.95
    deg_free = trial - 1

    for i in range(trial):
        # get path
        data_path_temp = data_path + '/hal/' + 'trial=' + str(i + 1)
        data_path_detail, _ = get_path_CQR(data_path=data_path_temp, method=method, loss=loss, gamma=gamma)

        result = np.load(data_path_detail)
        
        if i == 0:
            coverage = result['coverage'] / trial
            coverage_all = result['coverage_all'] / trial
            coverage_db = result['coverage_db'] / trial
            range_ave = result['range_ave'] / trial
            input_neo[0,:] = result['input_te'].reshape(-1)

        else:
            coverage += result['coverage'] / trial        
            coverage_all += result['coverage_all'] / trial
            coverage_db += result['coverage_db'] / trial
            range_ave += result['range_ave'] / trial
            input_neo[i,:] = result['input_te'].reshape(-1)
        
        cov_db_temp[i] = result['coverage_db'][:, -1].reshape(-1) 
        cov_temp[i] = (((result['coverage'])[1] - (result['coverage'])[0]).reshape(-1))[-1]

    cov_ave = st.tmean(cov_temp)
    cov_scale = np.sqrt(st.tvar(cov_temp) / trial)

    for j in range(3):
        cov_ave_db = st.tmean(cov_db_temp[:, j])
        cov_scale_db = np.sqrt(st.tvar(cov_db_temp[:, j]) / trial)
        cov_db_interval[j] = st.t.interval(alpha_rel, deg_free, loc=cov_ave_db, scale=cov_scale_db)
    
    cov_interval = st.t.interval(alpha_rel, deg_free, loc=cov_ave, scale=cov_scale)
    # save
    save_path, save_path_add = get_path_CQR(data_path=data_path, method=method, loss=loss, gamma=gamma)
    data_path = data_path.replace('alpha=0.95', 'alpha=0.95/hal')
    save_path = save_path.replace('alpha=0.95/', 'alpha=0.95/hal/')
    print(save_path)
    mkdir(data_path+save_path_add, exist_ok=True)
    np.savez_compressed(save_path, coverage=coverage, coverage_all=coverage_all, coverage_interval=cov_interval, coverage_db=10 * np.log10(coverage_db), coverage_db_interval=(10 * np.log10(cov_db_interval)), range_ave=range_ave, input_te=input_neo)

if __name__ == '__main__':
    for alpha_range_temp in range(config.limit):
        alpha_range = np.round(0.95 - (alpha_range_temp * 0.05), 3)
    
        data_path =  'result/text/dim=' + str(config.input_dim) + '/' + str(config.noise_type) + '/' + str(config.outlier_type) + '/Iter=' + str(config.Iter) + '/alpha=' + str(alpha_range)

        for _, method in enumerate(config.methods):
            data_integrate(data_path=data_path, method=method, loss=config.loss, gamma=config.gamma_default, trial=config.trial, input_te=input_te)