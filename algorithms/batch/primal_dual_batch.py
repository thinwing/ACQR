import numpy as np
from scipy.sparse.linalg import eigs
import sys
sys.path.append('.../')
from configuration import graph_config as grp
from configuration import config

def pinball_moreau(x, alpha, gamma):
    # x is vector
    x = x / gamma
    temp = np.where(x >= alpha, alpha, x)
    result = np.where(temp <= (alpha - 1), alpha - 1, temp)

    return result

def pinballized_huber(x, alpha, gamma):
    # x = grad / delta
    # x is vector
    x = x / gamma

    for index, element in enumerate(x):
        if element >= 1:
            x[index] = alpha
        
        elif element >= 0:
            x[index] = alpha * element

        elif element >= -1:
            x[index] = (1 - alpha) * element
        
        else:
            x[index] = (1 - alpha) * (-1)

    result = x

    return result

def pinballized_square(x, alpha):
    # x is vector
    # result = np.where(x > 0, alpha * x, (1 - alpha) * x)
    result = x

    return result.reshape(-1, 1)
  
def biased_threshold(x, alpha, gamma):
    # gamma = mu / sigma
    prox_pinball = x - (gamma * pinball_moreau(x, alpha, gamma))

    return prox_pinball


def updating(output, v_temp, kernel_matrix, step_size, regular, alpha, kernel_weight, gamma):    
    # the size of kernel matrix is [r, n] (r is the dictionary size and n is the number of outputs.)

    # Step 1
    grad_temp = kernel_weight - (step_size[0] * (np.matmul(kernel_matrix, pinballized_square(kernel_weight, alpha)) + (regular * np.matmul(kernel_matrix, pinballized_huber(output - np.matmul(kernel_matrix.T, kernel_weight), alpha, gamma))) ))
    # grad_temp = kernel_weight - (step_size[0] * ( pinballized_square(kernel_weight, alpha) - (regular * np.matmul(kernel_matrix, pinballized_huber(np.matmul(kernel_matrix.T - output, kernel_weight), alpha, gamma))) ))

    # Step 2
    prox_temp = grad_temp + (step_size[0] * np.matmul(kernel_matrix, v_temp))
    # prox_temp = grad_temp - (step_size[0] * np.matmul(kernel_matrix, v_temp))

    # Step 3
    temp = v_temp + (step_size[1] * (output - np.matmul(kernel_matrix.T, prox_temp)))
    # temp = v_temp - (step_size[1] * (output - np.matmul(kernel_matrix.T, prox_temp)))

    prox = temp - (step_size[1] * biased_threshold((temp / step_size[1]), alpha, (regular / step_size[1]))) 

    # Step 4
    grad = grad_temp + (step_size[0] * np.matmul(kernel_matrix, prox))
    # grad = grad_temp - (step_size[0] * np.matmul(kernel_matrix, prox))

    # updating
    kernel_weight_update = kernel_weight + (step_size[2] * (grad - kernel_weight))
    v = v_temp + (step_size[2] * (prox - v_temp))

    return kernel_weight_update, v

class batch_learning_primal():
    def __init__(self, alpha, loss, Iter=int, kernel_vector=np.array, kernel_vector_eval=np.array, output_train=np.array):
        self.loss = loss
        self.alpha = alpha
        self.Iter = Iter
        self.kernel_vector = kernel_vector
        self.kernel_vector_eval = kernel_vector_eval
        self.output_train = output_train

    def learning(self, step_size, regular):
        # step_size = np.array([tau, sigma, beta])
        kernel_weight = np.zeros([len(self.alpha), len(self.kernel_vector), 1])
        kernel_func = np.zeros([len(self.alpha), self.Iter, len(self.output_train)])
        v_temp = np.zeros([len(self.kernel_vector[0]), 1])

        # tau
        if regular == 0:
            temp, _ = eigs(self.kernel_vector, 1)
            regular = self.loss['gamma'] / (abs(temp))
            print(regular)
            # regular = 0.0001

        if step_size[0] == 0:
            temp, _= eigs(self.kernel_vector.T @ self.kernel_vector, 1)
            temp = abs(temp)
            step_size[0] = 2 / (1 + (temp * regular / self.loss['gamma']))
            step_size[1] = 1 / (step_size[0] * temp)


        for i in range(self.Iter):
            for alpha_index, alpha in enumerate(self.alpha):
                kernel_weight[alpha_index], v_temp = updating(output=self.output_train, v_temp=v_temp, kernel_matrix=self.kernel_vector, step_size=step_size, regular=regular, alpha=alpha, kernel_weight=kernel_weight[alpha_index], gamma=self.loss['gamma'])
                kernel_func[alpha_index, i] = np.matmul(self.kernel_vector_eval.T, kernel_weight[alpha_index]).reshape(-1)


        return kernel_func



        