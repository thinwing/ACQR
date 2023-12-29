import numpy as np

import sys
sys.path.append('.../')
from configuration import graph_config as grp
from configuration import config

def pinball_moreau(x, alpha, gamma):
    x = x / gamma
        
    if x > alpha:
        result = alpha
    
    elif x < alpha - 1:
        result = alpha - 1
    
    else:
        result = x

    return result

def pinballized_huber(x, alpha, gamma):
    # x = grad / delta
    x = x / gamma

    if x >= 1:
        result = alpha
    
    elif x <= -1:
        result = alpha - 1
    
    elif x < 0:
        result = (1 - alpha) * x
    
    else:
        result = alpha * x

    return result

def pinballized_square(x, alpha):
    # x is vector
    result = np.where(x > 0, alpha * x, (1 - alpha) * x)

    return result.reshape(-1, 1)

def biased_threshold(x, alpha, gamma):
    # gamma = mu / sigma
    prox_pinball = x - (gamma * pinball_moreau(x, alpha, gamma))

    return prox_pinball


def updating(output, v_temp, kernel_vector, step_size, regular, alpha, kernel_weight, gamma):    
    # Step 1
    kernel_vector = kernel_vector.reshape(-1, 1)
    grad_temp = kernel_weight - (step_size[0] * ( pinballized_square(kernel_weight, alpha) - (regular * kernel_vector * pinballized_huber(np.dot(kernel_weight.T, kernel_vector) - output, alpha, gamma)) ))


    # Step 2
    prox_temp = grad_temp - (step_size[0] * v_temp * kernel_vector)

    # Step 3
    temp = v_temp + (step_size[1] * np.dot(kernel_vector.T, prox_temp))
    # print(temp)

    prox = temp - (step_size[1] * ( output + biased_threshold(temp / step_size[1], alpha, (regular / step_size[1])))) 

    # Step 4
    grad = grad_temp - (step_size[0] * prox * kernel_vector)

    # updating
    kernel_weight_update = kernel_weight + (step_size[2] * (grad - kernel_weight))
    v = v_temp + (step_size[2] * (prox - v_temp))

    return kernel_weight_update, v

class online_learning_primal():
    def __init__(self, alpha, loss, Iter=int, kernel_vector=np.array, kernel_vector_eval=np.array, output_train=np.array):
        self.loss = loss
        self.alpha = alpha
        self.Iter = len(kernel_vector[0])
        self.kernel_vector = kernel_vector
        self.kernel_vector_eval = kernel_vector_eval
        self.output_train = output_train

    def learning(self, step_size, regular):
        # step_size = np.array([tau, sigma, beta])
        kernel_weight = np.zeros([len(self.alpha), len(self.kernel_vector), 1])
        kernel_func = np.zeros([len(self.alpha), self.Iter, len(self.output_train)])
        v_temp = np.zeros(1)

        for i in range(self.Iter):
            for alpha_index, alpha in enumerate(self.alpha):
                kernel_weight[alpha_index], v_temp = updating(output=self.output_train[i], v_temp=v_temp, kernel_vector=self.kernel_vector[:, i], step_size=step_size, regular=regular, alpha=alpha, kernel_weight=kernel_weight[alpha_index], gamma=self.loss['gamma'])
                kernel_func[alpha_index, i] = np.dot(kernel_weight[alpha_index].T, self.kernel_vector_eval).reshape(-1)

        return kernel_func



        