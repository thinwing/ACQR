import numpy as np
from data_loader import *
from os import makedirs as mkdir
import configuration.config as config

data_path = 'None'

def dt(data_path=data_path, Iter=config.Iter, input_dim=config.input_dim, noise_type=config.noise_type, outlier_type=config.outlier_type, outlier_rate=config.outlier_rate):
    mkdir(data_path, exist_ok=True)
    data_path += '/'
    # data create > save as npz file
    input_train, output_true_train, input_test, output_true_test = dt_data(Iter=Iter, input_dim=input_dim)
    
    np.savez_compressed(data_path + 'data.npz', input_train=input_train, output_true_train=output_true_train, input_test=input_test, output_true_test=output_true_test)
     
    # noise create > save as npz file
    # outlier create > save as npz file
    
    noise_train, noise_real_train = dt_noise(input=input_train, noise_type=noise_type)
    # Frozen
    # noise_test, noise_real_test = dt_noise(input=input_test, noise_type=noise_type)
    noise_test = noise_train
    noise_real_test = noise_real_train 
    
    np.savez_compressed(data_path + 'noise.npz', noise_train=noise_train, noise_real_train=noise_real_train, noise_test=noise_test, noise_real_test=noise_real_test, noise_type=noise_type)
 
    output_train, output_train_noise = dt_outlier(input=input_train, output_true=output_true_train, noise_real=noise_real_train, outlier_type=outlier_type, outlier_rate=outlier_rate)
    # Frozen
    # output_test = dt_outlier(input=input_test, output_true=output_true_test, noise_real=noise_real_test, outlier_type=outlier_type, outlier_rate=outlier_rate)               
    output_test = output_train
    output_test_noise = output_train_noise
    
    np.savez_compressed(data_path + 'outlier.npz', output_train=output_train, output_test=output_test, output_train_noise=output_train_noise, output_test_noise=output_test_noise, outlier_type=outlier_type)

#def dt_c(data_path=data_path, Iter=config.Iter, input_dim=config.input_dim, noise_type=config.noise_type, outlier_type=config.outlier_type, outlier_rate=config.outlier_rate):
    #data_path += '/calib'    
    #mkdir(data_path, exist_ok=True)
    # data create > save as npz file
    #input_train_c, output_true_train_c, input_test_c, output_true_test_c = dt_calib(Iter=Iter, input_dim=input_dim)
    
    #np.savez_compressed(data_path + '/data_c', input_train=input_train_c, output_true_train=output_true_train_c, input_test=input_test_c, output_true_test=output_true_test_c)
     
    # noise create > save as npz file
    # outlier create > save as npz file
    
    #noise_train_c, noise_real_train_c = dt_noise(input=input_train_c, noise_type=noise_type)
    # Frozen
    # noise_test, noise_real_test = dt_noise(input=input_test, noise_type=noise_type)
    #noise_test_c = noise_train_c
    #noise_real_test_c = noise_real_train_c 
    
    #np.savez_compressed(data_path + '/noise_c', noise_train=noise_train_c, noise_real_train=noise_real_train_c, noise_test=noise_test_c, noise_real_test=noise_real_test_c, noise_type=noise_type)
 
    #output_train_c, output_train_noise_c = dt_outlier(input=input_train_c, output_true=output_true_train_c, noise_real=noise_real_train_c, outlier_type=outlier_type, outlier_rate=outlier_rate)
    # Frozen
    # output_test = dt_outlier(input=input_test, output_true=output_true_test, noise_real=noise_real_test, outlier_type=outlier_type, outlier_rate=outlier_rate)               
    #output_test_c = output_train_c
    #output_test_noise_c = output_train_noise_c
    
    #np.savez_compressed(data_path + '/outlier_c', output_train=output_train_c, output_test=output_test_c, output_train_noise=output_train_noise_c, output_test_noise=output_test_noise_c, outlier_type=outlier_type)
    
if __name__ == '__main__':
    #for i in range(11, config.trial):
        i = 10
        data_path = 'exp_data/' + 'dim=' + str(config.input_dim) + '/' + str(config.noise_type) + '/' + str(config.outlier_type) + '/Iter=' + str(config.Iter) + '/trial=' + str(i+1)
        dt(data_path=data_path)