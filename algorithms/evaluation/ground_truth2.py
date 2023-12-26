import numpy as np
import math 
from scipy.stats import norm

def groundandsame2(output_true_test, output_test, noise, alpha, out):
    temp = norm.ppf(1.0)
    alpha_range = 1.0
    output_true_test = np.delete(out, output_true_test)
    output_test = np.delete(out, output_test)
    base = 0.001
    clock = 0
    range = 0

    lower = output_true_test
    upper = output_true_test
    
    while clock < 10:
        range += base * 2
        upper = upper + base
        lower = lower - base

        coverage_temp = np.where((upper >= output_test) & (lower <= output_test), 1, 0)
        coverage = np.sum(coverage_temp) / len(coverage_temp)

        if coverage >= alpha_range:
            clock = 10000 

    same_range = np.ones([len(output_test)]) * range
     
    range_gt = np.copy(noise) * 2 * temp
    range_gt_ave = np.average(range_gt) * np.ones([len(output_test)])
    
    sr = np.vstack((lower.T, upper.T)) 
    
    gt_low = output_true_test - (range_gt / 2)
    gt_high = output_true_test + (range_gt / 2)
    
    gt = np.vstack((gt_low.T, gt_high.T))
    
    return sr, gt, same_range, range_gt_ave