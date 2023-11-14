import numpy as np
import csv
from itertools import zip_longest

a = np.random.rand()
b = np.random.rand()

def toy_data(input, center):
    bandwidth = np.array([0.1, 0.3])
    output_true = 4 * (a + 0.1) * np.exp((-1) * np.sum((input - center[0]) ** 2, axis=1) / (2 * (bandwidth[0] ** 2))) + 2 * (b + 0.1) * np.exp((-1) * np.sum((input - center[1]) ** 2, axis=1) / (2 * (bandwidth[1] ** 2)))
    output_true = output_true.reshape(-1, 1) 
        
    return output_true

def toy_datab(input, center):
    bandwidth = np.array([0.1, 0.3])
    shift = np.array([0.1])
    output_true = 4 * (a + 0.1) * np.exp((-1) * np.sum((input + shift - center[0]) ** 2, axis=1) / (2 * (bandwidth[0] ** 2))) + 2 * (b + 0.1) * np.exp((-1) * np.sum((input + shift - center[1]) ** 2, axis=1) / (2 * (bandwidth[1] ** 2)))
    output_true = output_true.reshape(-1, 1) 
        
    return output_true

def dt_create(Iter, input_dim):
    # input : dim 2
    t = 0
    # All point
    while t < 1:
        input_temp = np.random.rand(Iter, input_dim)
        input_unique = np.unique(input_temp, axis=0)

        if len(input_temp) == len(input_unique):
            t = 10
            input_train = input_temp
            
        else:
            t = 0 

    if input_dim == 1:
        center = np.array([0.2, 0.6])
    else:
        center = np.random.rand(2, input_dim)

    inputa, inputb = np.array_split(input_train, 2, 0)
    # output
    outputa = toy_data(inputa, center)

    outputb = toy_datab(inputb, center)

    output_train = np.vstack((outputa,outputb))

    input_test = input_train

    output_test = output_train 
    
    with open('input.csv', 'w') as f:
        writer = csv.writer(f, lineterminator='\n')
        for i in input_train:
            writer.writerow([i])

    with open('output.csv', 'w') as f:
        writer = csv.writer(f, lineterminator='\n')
        for i in output_train:
            writer.writerow([i])

    return input_train, output_train, input_test, output_test