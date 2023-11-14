import numpy as np
import configuration.config as config

# Methods
## ground truth
same_range = {'save_name':'same_range'}
ground_truth = {'save_name':'ground_truth'}

## Batch
NN = {'method':'QRNN', 'save_name':'QRNeuralNetwork', 'processing':'batch'}
QRF = {'method':'QRF', 'save_name':'QuantileRandomForest', 'processing':'batch'}
KQR = {'method':'KQR', 'save_name':'KernelQR', 'processing':'batch'}

## Online
single_kernel = {'method':'Kernel', 'variable':config.coherence, 'dict_band':config.dict_band_single, 'save_name':'single_kernel', 'processing':'online'}
single_rff = {'method':'RFF', 'variable':config.num_rff, 'dict_band':config.dict_band_single, 'save_name':'single_rff', 'processing':'online'}

multi_kernel = {'method':'Kernel', 'variable':config.coherence, 'dict_band':config.dict_band_multi, 'save_name':'multi_kernel', 'processing':'online'}
multi_rff = {'method':'RFF', 'variable':config.num_rff, 'dict_band':config.dict_band_multi, 'save_name':'multi_rff', 'processing':'online'}


# Loss function
pinball = {'loss':'pinball', 'gamma':0}
pinball_moreau = {'loss':'pinball_moreau', 'gamma':0.5}
pmc_online = {'loss':'pmc_online', 'gamma':0.5}
pmc_batch = {'loss':'pmc_batch', 'gamma':0.5}

pinball_smooth_relax = {'loss':'pinball_smooth_relax', 'gamma':0.4}
pinball_huberized = {'loss':'pinball_huberized', 'gamma':0.5}

