import numpy as np
import configuration.config as config

range_alpha = np.array([0.95])

loss_base = 'pinball_moreau'
loss_all = ['pinball','pinball_moreau','pinball_huberized','pinball_smooth_relax', 'pmc_batch']

# Methods
## ground truth
same_range = {'save_name':'same_range', 'loss':'None', 'address':'base', 'fig_name':'same range', 'color':'black',}
ground_truth = {'save_name':'ground_truth', 'loss':'None', 'address':'base', 'fig_name':'ground truth', 'color':'black'}

## Batch
NN = {'save_name':'QRNeuralNetwork', 'loss':'None', 'address':'batch', 'fig_name':'QRNN [10]', 'color':'green', 'marker':'o', 'alpha':1}
QRF = {'save_name':'QuantileRandomForest', 'loss':'None', 'address':'batch', 'fig_name':'QRF [2]', 'color':'gold', 'marker':'s', 'alpha':0.4}
KQR = {'save_name':'KernelQR', 'loss':'None', 'address':'batch', 'fig_name':'KQR [12]', 'color':'purple', 'marker':'s', 'alpha':1}

## Online
single_kernel = {'save_name':'single_kernel', 'address':'online', 'loss':loss_base, 'fig_name':'proposed 1 (single)', 'color':'red', 'marker':'v', 'alpha':1, 'fig_name2':'(single)'}
single_rff = {'save_name':'single_rff', 'address':'online', 'loss':loss_base, 'fig_name':'proposed 3 (single RFF)', 'color':'deeppink', 'marker':'v', 'fig_name2':'(single RFF)'}

multi_kernel = {'save_name':'multi_kernel', 'address':'online', 'loss':loss_base, 'fig_name':'proposed 2 (multiple)', 'color':'blue', 'marker':'^', 'alpha':1, 'fig_name2':'(multiple)'}
multi_rff = {'save_name':'multi_rff', 'address':'online', 'loss':loss_base, 'fig_name':'proposed 4 (multi RFF)', 'color':'aqua', 'marker':'^', 'fig_name2':'(multi RFF)'}



# Loss function
pinball = {'loss':'pinball', 'gamma':0, 'loss_name':'pinball', 'marker':'v', 'color':['green', 'lime']}
pinball_moreau = {'loss':'pinball_moreau', 'gamma':config.gamma_default, 'loss_name':'proposed', 'marker':'^', 'color':['red','orange']}
pinball_smooth_relax = {'loss':'pinball_smooth_relax', 'gamma':0.1, 'loss_name':'smooth', 'marker':'v', 'color':['navy', 'aqua']}
pinball_huberized = {'loss':'pinball_huberized', 'gamma':config.gamma_default, 'loss_name':'huberized pinball', 'marker':'v', 'color':'lime'}

pmc_online = {'loss':'pmc_online', 'gamma':0.5, 'loss_name':'Pinballized MC', 'marker':'v', 'color':['navy', 'aqua']}
pmc_batch = {'loss':'pmc_batch', 'gamma':50.0, 'loss_name':'Pinballized MC', 'marker':'v', 'color':['navy', 'aqua']}


loss_list = np.array(['pinball_moreau'])

# list for graph
list_graph = np.array(['single_kernel'])
list_graph_online = np.array(['single_kernel'])

list_graph_coverage = np.array(['single_kernel'])


# graph
## figure_observation
fig_size_observation_base = np.array([12, 8])
linewidth = 3
font_size = 36
ticks = 36
dot_size = 6


## figure_coverage


## figure_alpha_coverage
marker_size = 16
title_coverage_db = ('range_error_alpha.png','error_low_alpha.png','error_high_alpha.png',)
#title_outlier_coverage_db = ('range_error_outlier.png','error_low_outlier.png','error_high_outlier.png',)
title_outlier_coverage_db = ('range_error_outlier.pdf','error_low_outlier.pdf','error_high_outlier.pdf',)
title_db = ('Range error [dB]', 'Quantile estimation error [dB]', 'Quantile estimation error [dB]')
#title_coverage_db_gamma = ('range_error_gamma.png','error_low_gamma.png','error_high_gamma.png',)
title_coverage_db_gamma = ('range_error_gamma.pdf','error_low_gamma.pdf','error_high_gamma.pdf',)

## figure_online


## figure_range