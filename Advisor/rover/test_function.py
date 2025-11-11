from openbox.utils.config_space.util import convert_configurations_to_array
import numpy as np

class mathFunction:
    def __init__(self, scale = 1.0, trans = 0, sft = 0):
        self.scale = scale
        self.trans = trans
        self.sft = sft
    
    def __call__(self, conf):
        x = conf
        return self.scale * self.calculate(x + self.sft) + self.trans
    
    def calculate(self, x):
        return NotImplementedError()
    
class Hartmann_3(mathFunction):
    def __init__(self, scale = 1.0, trans = 0, sft = 0):
        super().__init__(scale, trans, sft)
        self.alpha = np.array([1.0, 1.2, 3.0, 3.2])
        self.A = np.array([[3.0, 10, 30], 
                           [0.1, 10, 35], 
                           [3.0, 10, 30],
                           [0.1, 10, 35]])
        self.P = 1e-4 * np.array([[3689, 1170, 2673], 
                                  [4699, 4387, 7470], 
                                  [1091, 8732, 5547], 
                                  [381,  5743, 8828]])

    def calculate(self, x):
        res = 0
        for i in range(4):
            s = 0
            for j in range(3):
                s += self.A[i, j] * ((x[j] - self.P[i, j]) ** 2)
            res += self.alpha[i] * np.exp(-s)
        return -res
    
    def get_min(self):
        return -3.86278 * self.scale + self.trans