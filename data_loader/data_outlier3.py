import numpy as np
import configuration.config as config

def outlier_create_lo(input, output_true, noise_real, outlier_type):
    output = output_true + noise_real
    
    output_noise = output
    Iter = config.Iter
    size = int(0.05*Iter)
        
    if outlier_type == 'sparse':
        arr = np.random.choice(Iter, size=size, replace=False)
        zero = np.zeros((4500, 1))
        for s in arr:
            zero[s] = -1 * np.random.randint(10, 101)
        print(zero[s])    
        output = output + zero
        
    elif outlier_type == 'impulse':
        sparse_vector = np.zeros([len(input), 1])
        for i in range(len(input)):
            if (i + 1) % 20 == 0:
                sparse_vector[i] = 1
        output = output + (np.sqrt(1000) * np.random.randn(len(input), 1) * sparse_vector)
        
    return output, output_noise

def outlier_create_hi(input, output_true, noise_real, outlier_type):
    output = output_true + noise_real
    
    output_noise = output
    Iter = config.Iter
    size = int(0.05*Iter)
        
    if outlier_type == 'sparse':
        arr = np.random.choice(Iter, size=size, replace=False)
        zero = np.zeros((4500, 1))
        for s in arr:
            zero[s] = np.random.randint(10, 101)
        print(zero[s])    
        output = output + zero
        
    elif outlier_type == 'impulse':
        sparse_vector = np.zeros([len(input), 1])
        for i in range(len(input)):
            if (i + 1) % 20 == 0:
                sparse_vector[i] = 1
        output = output + (np.sqrt(1000) * np.random.randn(len(input), 1) * sparse_vector)
        
    return output, output_noise

def outlier_create_hal(input, output_true, noise_real, outlier_type):
    output = output_true + noise_real
    
    output_noise = output
    Iter = config.Iter
    size = int(0.05*Iter)
        
    if outlier_type == 'sparse':
        arr = np.random.choice(Iter, size=size, replace=False)
        arr1, arr2 = np.array_split(arr, 2, 0)
        zero = np.zeros((4500, 1))
        for s in arr1:
            zero[s] =  np.random.randint(10, 101)
        for t in arr2:
            zero[t] =  -1 * np.random.randint(10, 101)
        print(zero[s])    
        output = output + zero
        
    elif outlier_type == 'impulse':
        sparse_vector = np.zeros([len(input), 1])
        for i in range(len(input)):
            if (i + 1) % 20 == 0:
                sparse_vector[i] = 1
        output = output + (np.sqrt(1000) * np.random.randn(len(input), 1) * sparse_vector)
        
    return output, output_noise