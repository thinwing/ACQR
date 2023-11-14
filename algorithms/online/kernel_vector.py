import numpy as np

class Kernel():
    def __init__(self, input, dict_band):
        self.dict_band = dict_band
        self.multi = len(dict_band)
        self.input = input
        self.input_num = len(input)
        self.input_dim = len(input[0])

    def dict_define(self, coherence):
        if self.input_dim == 1:
            input_num = self.input_num
            input = self.input
        else:    
            input_num = 200
            input = np.linspace(0, 1, num=input_num).reshape(input_num, self.input_dim)

        self.dict = input[0].reshape(1, self.input_dim)
        
        for j in range(input_num - 1):
            max_coherence = 0
            for q in range(self.multi):
                coherence_temp = np.exp((-1) * np.sum((self.dict - input[j + 1]) ** 2, axis=1) / (2 * (self.dict_band[q] ** 2)))

                max_coherence_new = np.max(coherence_temp)
                
                if max_coherence_new > max_coherence:
                    max_coherence = max_coherence_new
            
            if max_coherence < coherence:
                add_temp = self.input[j + 1].reshape(1, self.input_dim)
                self.dict = np.array(np.vstack((self.dict, add_temp)))
                self.dict = self.dict.reshape(-1, self.input_dim)
    
        self.dict_num = len(self.dict)
        print(self.dict_num)

        # print('Dictionary Size = ' + str(self.dict_num * self.multi) )

    def kernel_vector(self, input):
        # all kernel vectors

        kernel_vector_test = np.zeros([(self.dict_num * self.multi), len(input)])
        for i in range(len(input)):
            for q in range(self.multi):
                kernel_vector_test[(self.dict_num * q) : (self.dict_num * (q + 1)), i] = np.exp((-1) * np.sum((self.dict - input[i]) ** 2, axis=1) / (2 * (self.dict_band[q] ** 2)))

        return kernel_vector_test

    
class RFF(Kernel):   
    def dict_define(self, num_rff):
        # omega
        self.dict_size = self.multi * num_rff
        self.num_rff = num_rff
        self.dict = np.zeros([self.dict_size, (self.input_dim + 1)])
        
        if self.input_dim > 1:
            for i in range(self.multi):
                self.dict[self.num_rff * i : self.num_rff * (i + 1), 0 : self.input_dim] = (1 / self.dict_band[i]) * np.random.randn(self.num_rff, self.input_dim)
        elif self.input_dim == 1:
            for i in range(self.multi):
                self.dict[self.num_rff * i : self.num_rff * (i + 1), 0] = (1 / self.dict_band[i]) * np.random.randn(self.num_rff)
        else:
            print('The dimension of input is error')
        
        # bias
        pi = np.pi
        self.dict[0:self.dict_size, self.input_dim] = np.random.rand(self.dict_size) * 2 * pi
        
        # print('Dictionary Size = ' + str(self.dict_size))
        # print(num_rff)

    def kernel_vector(self, input):
        kernel_vector_test = np.zeros([self.dict_size, len(input)])
    
        if self.input_dim > 1:
            for i in range(len(input)):
                kernel_vector_test[:, i] = np.sqrt(2 / self.num_rff) * np.cos(np.dot(self.dict[:,0:self.input_dim], input[i]) + self.dict[:, -1])
        elif self.input_dim == 1:
            for i in range(len(input)):
                kernel_vector_test[:, i] = np.sqrt(2 / self.num_rff) * np.cos((self.dict[:,0] * input[i]) + self.dict[:, -1])

        return kernel_vector_test

    