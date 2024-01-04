import numpy as np
import os
#os.add_dll_directory(r"C:\Program Files\R\R-4.3.1\bin\x64")
#os.environ["R_HOME"]=r"C:\Program Files\R\R-4.3.1"
import rpy2.robjects as r2
import rpy2.robjects.numpy2ri as npr
from rpy2.robjects.packages import importr

import sys
sys.path.append('.../')
from configuration.config import *

class KQR():
    def __init__(self, alpha, input_train, output_train):
        self.alpha = alpha
        self.input_train = input_train
        self.output_train = output_train
        
        lab = importr('kernlab')
        self.kqr = r2.r.assign('kqr', lab.kqr)

        self.prd = r2.r['predict']
        self.qr = []
        
        self.regular = regular
        # RBF kernel
        self.sigma_rbf = 1 / ((sigma_rbf ** 2) * 2)
        
        # python -> R
        self.sigma = r2.r.assign("sg", self.sigma_rbf)
        self.list_R = r2.r("list(sigma=sg)")

    def pre_learning(self):
        npr.activate()
        for _, alpha_temp in enumerate(self.alpha):
            self.qr.append(self.kqr(self.input_train, self.output_train, kernel="rbfdot", kpar=self.list_R, tau=alpha_temp, C=self.regular))

    def predict(self, input_test):
        # predict
        npr.activate()
        y_est_low = self.prd(self.qr[0], input_test)
        y_est_high = self.prd(self.qr[1], input_test)

        result = np.vstack((y_est_low.T, y_est_high.T)).reshape(2, -1)
        
        return result
