import numpy as np
### THIS IS THE CONFIG FILE ###

optimize_flag = 'custom'
data_flag = 'off'
onlyCOQKRflag = 'off'
onlyACIflag = 'off'
# num_divide = 3

start = 0
limit = 1

#---------- data ----------#
input_dim = 1
output_dim = 1
# Iter = 20000
Iter = 3000
Iter_CQR = int(Iter/3)# figure
#Iter_a = 2000
#Iter_b = 1000
Iter_batch = 500
trial = 25
outlier_rate_temp = np.arange(6) * 0.02
#outlier_rate = outlier_rate_temp[:]
outlier_rate = np.array([0.05])

noise_type_all = ('normal', 'linear_expansion', 'exp_wave',)
noise_types = ('linear_expansion',)

outlier_type_all = ('nothing', 'sparse', 'impulse',)
outlier_types = ('sparse',)

#---------- type ------------#
## if you want to a training data, they are used.

noise_type = 'linear_expansion'
outlier_type = 'sparse'
outlier_rate_single = 0.05

#--------- algorithms ---------#
num_estimator = 1000
max_depth = 10
#num_split = 5
num_split = 0

## Quantile Regression Neural Network
num_hidden_layer = 64
lr = 0.0003
dropout = 0.2
activation = 'relu'

epochs = 200
validation = 0.2

## Kernel Quantile Regression
regular = 1
sigma_rbf = 0.1

## Gradient methods

#step_size = 0.005

#↓これ良さげ
#step_size = 0.05

#step_size = 0.1

#↓これ一番いい
#step_size = 0.0005

#↓CQR1はこれがよさげ？
#step_size = 0.001

#5％の場合は0.00付近
#0.00は区間が短い
#0.は区間が長い

#↓従来手法強すぎ伝説
#step_size = 0.00025

#↓CQR1ではこれが一番よさそう
step_size = 0.0003
coherence = 10.0
num_rff = 50 # no meaning

## Primal dual method
### parameter_pdm = np.array([tau, sigma, beta])
parameter_pdm = np.array([0.0, 0.0, 1])

### regular_pdm -> the larger it is, the more robust.
regular_pdm = np.array([0.0])


## Pinball Loss : Moreau Envelope
# gamma = 0.0001
# gamma = 100.0
gamma = 10.0 ** (np.arange(5) - 1)
#gamma = np.array([100.0, 1000.0])
# gamma_conditional = 0.1
gamma_default = 10.0
#gamma_default = 0.5

dict_band_single = np.array([0.1])
dict_band_multi = np.array([0.05, 0.1])

alpha_range_min = 0.05
alpha_base = (np.arange(int(1 / alpha_range_min) - 1) + 1) * alpha_range_min / 2
alpha_all = np.round(np.vstack((alpha_base, 1 - alpha_base)).T, 4)

range_alpha = np.round(alpha_all[:, 1] - alpha_all[:, 0], 3)

# alpha = np.array([0.05, 0.95])
alpha_range = 0.95

# methods
# 'NN', 'QRF', 'single_kernel', 'single_rff', 'multi_kernel', 'multi_rff'
method_all = ('NN', 'QRF', 'KQR', 'single_kernel', 'single_rff', 'multi_kernel', 'multi_rff',)
method = 'single_kernel'
methods = ('single_kernel', 'multi_kernel')

# Loss function
loss_all = ('pinball', 'pinball_moreau', 'pinball_smooth_relax', 'pinball_huberized', 'pmc_online', 'pmc_batch')
loss = 'pinball_moreau'
losses = ('pinball_moreau',)