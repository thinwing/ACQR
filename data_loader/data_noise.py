import numpy as np
import math

def noise_create(input, noise_type):
    if len(input[0]) > 1:
        noise = np.sqrt(np.sum(input**2, axis=1)).reshape(-1, 1)
    else:
        noise = input


    if noise_type == 'linear_expansion':
        beta = 1
        # base self.noise + expansion
        noise = np.sqrt(0.3 + beta * noise)

    # elif noise_type == 'exp_expansion':
    #     beta = 1 
    #     delta = 1
    #     # base self.noise + expansion
    #     noise = np.sqrt(0.1 + delta * np.exp(beta * noise))

    # elif noise_type == 'exp_decay':
    #     beta = 10
    #     delta = 2
    #     # base self.noise + expansion
    #     noise = np.sqrt(0.1 + delta * (1 - np.exp(-1 * beta * noise)))

    elif noise_type == 'exp_wave':
        omega = 1
        omega = omega * 2 * math.pi
        delta = 1

        noise = np.sqrt(0.3 + delta * np.exp(np.sin(omega * noise)))
        
    # elif noise_type == 'rectangle':
    #     num_mount = 2
    #     delta = 2
        
    #     point_zero = num_mount * 2 
    #     delta = delta * 2
        
    #     noise = noise.reshape(-1)
        
    #     for i in range(point_zero):
    #         if i % 2 == 0:
    #             temp_flag = 100
    #         else:
    #             temp_flag = 0
            
    #         noise = np.where((((1 / point_zero) * i) <= noise) & (noise <= ((1 / point_zero) * (i + 1))), temp_flag, noise)
        
    #     noise = noise.reshape(-1, 1) / 100
    #     noise = np.sqrt(0.1 + (noise * delta))
    
    else:
        noise = np.sqrt(0.3) * np.ones([len(input), 1])
    
    noise_real = noise * np.random.randn(len(input), 1)
    
    return noise, noise_real
