import numpy as np

def cov(func_est, output_test, alpha, Iter, method):
    # PINBALL
    coverage = np.zeros([len(alpha), Iter])

    if method['processing'] == 'online':
        for i in range(len(alpha)):
            print('func_est')
            print(func_est)
            print(output_test.T)
            print(func_est[i] - output_test.T)
            coverage_temp = np.where((func_est[i] - output_test.T > 0), 1, 0)
            coverage[i] = np.sum(coverage_temp, axis=1) / len(coverage_temp[0])
            #ここにfor文で繰り返しさせれば行けそうじゃない？
    else:
        for i in range(len(alpha)):
            coverage_temp = np.where((func_est[i] - output_test.reshape(-1) > 0), 1, 0)
            coverage[i] = np.sum(coverage_temp, axis=0) / len(coverage_temp) * np.ones(Iter)

    return coverage