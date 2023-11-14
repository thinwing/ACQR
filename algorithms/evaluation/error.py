import numpy as np

def range_error(func_est, gt, Iter, method):
    range_func_est = func_est[1] - func_est[0]

    range_gt = gt[1] - gt[0]

    if method['processing'] == 'online':
        range_func_est_ave = np.sum(range_func_est, axis=1) / len(range_func_est[0])

        coverage_db_temp = np.sum((range_func_est - range_gt.T) ** 2, axis=1) / np.sum(range_gt ** 2, axis=0)  
    
        coverage_db_high = np.sum((func_est[1] - gt[1].T) ** 2, axis=1) / np.sum(gt[1] ** 2, axis=0)        
        coverage_db_low = np.sum((func_est[0] - gt[0].T) ** 2, axis=1) / np.sum(gt[0] ** 2, axis=0)

        coverage_db_temp = np.vstack((coverage_db_temp.T, coverage_db_low.T, coverage_db_high.T))
        
        
    else:
        range_func_est_ave = np.sum(range_func_est, axis=0) / len(range_func_est) * np.ones(Iter)

        coverage_db_temp = np.sum((range_func_est - range_gt) ** 2, axis=0)/ np.sum(range_gt ** 2, axis=0) * np.ones(Iter)

        coverage_db_high = np.sum((func_est[1] - gt[1].T) ** 2, axis=0) / np.sum(gt[1] ** 2, axis=0) * np.ones(Iter)
        coverage_db_low = np.sum((func_est[0] - gt[0].T) ** 2, axis=0) / np.sum(gt[0] ** 2, axis=0) * np.ones(Iter)
        
        coverage_db_temp = np.vstack((coverage_db_temp.T, coverage_db_low.T, coverage_db_high.T))
                
        
    return range_func_est_ave, coverage_db_temp