import numpy as np

class loss_func():
    def __init__(self, x, alpha):
        self.x = x
        self.alpha = alpha

    def pinball(self, gamma):
        if self.x > 0:
            result = self.alpha
            
        elif self.x < 0:
            result = self.alpha - 1
        
        else:
            result = self.x

        return result

    def pinball_moreau(self, gamma):
        x = self.x / gamma
        
        if x > self.alpha:
            result = self.alpha
        
        elif x < self.alpha - 1:
            result = self.alpha - 1
        
        else:
            result = x

        return result

    def pinball_smooth_relax(self, gamma):
        
        result = self.alpha - (1 / (1 + np.exp(self.x / gamma)))
        
        return result

    def pinball_huberized(self, gamma):
        # x = grad / delta
        x = self.x / gamma

        if x >= 1:
            result = self.alpha
        
        elif x <= -1:
            result = self.alpha - 1
        
        elif x < 0:
            result = (1 - self.alpha) * x
        
        else:
            result = self.alpha * x

        return result
    
def updating(output, kernel_vector, step_size, alpha, kernel_weight, loss):
    kernel_vector = kernel_vector.reshape(-1, 1)
    grad_temp = (output - np.dot(kernel_weight.T, kernel_vector))
    
    ls = loss_func(x=grad_temp, alpha=alpha)
    
    kernel_weight_update = step_size * eval('ls.' + loss['loss'])(loss['gamma']) * kernel_vector
    
    kernel_weight_new = kernel_weight + kernel_weight_update
    
    return kernel_weight_new