import numpy as np

def outlier_create2(input, output_true, noise_real, outlier_type):    
    output = output_true + noise_real
    
    output_noise = output
        
    if outlier_type == 'sparse':
        arr1 = np.arange(20, 2981, 40)
        arr2 = np.arange(40, 3001, 40)
        zero = np.zeros((3000, 1))
        for s in arr1:
            zero[s,1] = -1 * np.random.randint(10, 101)
        for t in arr2:
            zero[t,1] = 1 * np.random.randint(10, 101)        
        output = output + zero
        
    elif outlier_type == 'impulse':
        sparse_vector = np.zeros([len(input), 1])
        for i in range(len(input)):
            if (i + 1) % 20 == 0:
                sparse_vector[i] = 1
        output = output + (np.sqrt(1000) * np.random.randn(len(input), 1) * sparse_vector)
        
    return output, output_noise