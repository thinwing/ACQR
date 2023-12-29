import numpy as np

def outlier_create2(input, output_true, noise_real, outlier_type):    
    output = output_true + noise_real
    
    output_noise = output
        
    if outlier_type == 'sparse':
        arr1 = np.arange(19, 2980, 40)
        arr2 = np.arange(39, 3000, 40)
        #arr1 = np.arange(24, 2975, 50)
        #arr2 = np.arange(49, 3000, 50)
        #out = np.hstack((arr1, arr2))
        zero = np.zeros((3000, 1))
        for s in arr1:
            zero[s] = -1 * np.random.randint(10, 101)
        for t in arr2:
            zero[t] = np.random.randint(10, 101)    
        print(zero[s])    
        output = output + zero
        
    elif outlier_type == 'impulse':
        sparse_vector = np.zeros([len(input), 1])
        for i in range(len(input)):
            if (i + 1) % 20 == 0:
                sparse_vector[i] = 1
        output = output + (np.sqrt(1000) * np.random.randn(len(input), 1) * sparse_vector)
        
    return output, output_noise