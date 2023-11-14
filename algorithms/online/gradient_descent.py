import numpy as np
from .loss_function import updating

class online_learning():
    def __init__(self, alpha, loss, Iter, kernel_vector, kernel_vector_eval, output_train):
        self.alpha = alpha
        self.Iter = len(kernel_vector[0])
        self.loss = loss
        self.kernel_vector = kernel_vector
        self.kernel_vector_eval = kernel_vector_eval
        self.output_train = output_train
        
    def learning(self, step_size):
        kernel_weight = np.zeros([len(self.alpha), len(self.kernel_vector), 1])
        kernel_func = np.zeros([len(self.alpha), self.Iter, len(self.output_train)])
        
        for i in range(self.Iter):        
            # Pinball Moreau
            for a in range(len(self.alpha)):
                kernel_weight[a] = updating(output=self.output_train[i], kernel_vector=self.kernel_vector[:, i], step_size=step_size, alpha=self.alpha[a], kernel_weight=kernel_weight[a], loss=self.loss)
                kernel_func[a, i] = np.dot(kernel_weight[a].T, self.kernel_vector_eval).reshape(-1)
                
        return kernel_func

            