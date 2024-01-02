import graph
import data as dt
import optimize_ACI 
import datetime
from integrate import data_integrate as integrate
from ACI import runACI
import csv

import configuration.config as config
import configuration.graph_config as graph_config
import configuration.address as address

import numpy as np
# data 

path = r'exp_data\dim=1\linear_expansion\sparse\outlier_rate=0.05\Iter=3000\trial=15\outlier.npz'
data = np.load(path)
output = data['output_test']
with open('checkhi.csv', 'w') as f:
    writer = csv.writer(f, lineterminator='\n')
    for i in output:
        writer.writerow([i])
