import numpy as np

def outlier_create(input, output_true, noise_real, outlier_type, outlier_rate):
    if outlier_type == 'cauchy':
        output = output_true + np.random.standard_cauchy(len(input)).reshape(-1, 1)
    else:    
        output = output_true + noise_real
    
    output_noise = output
        
    if outlier_type == 'sparse':
        sparse_temp = np.random.rand(len(input), 1)
        sparse_vector = np.where(sparse_temp < outlier_rate, 1, 0)
        output = output + (np.sqrt(1000) * np.random.randn(len(input), 1)) * sparse_vector
        
    elif outlier_type == 'impulse':
        sparse_vector = np.zeros([len(input), 1])
        for i in range(len(input)):
            if (i + 1) % 20 == 0:
                sparse_vector[i] = 1
        output = output + (np.sqrt(1000) * np.random.randn(len(input), 1) * sparse_vector)
        
    return output, output_noise