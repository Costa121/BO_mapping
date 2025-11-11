import numpy as np
from openbox import logger

class Test_Function:
    def __init__(self, dim_effective, dim_tot):
        self.dim_e = dim_effective
        self.d = dim_tot
    
    def __call__(self, config):
        raise NotImplementedError

class Branin(Test_Function):
    def __init__(self, dim_effective, dim_tot):
        super().__init__(dim_effective, dim_tot)
        
        self.a = 1
        self.b = 5.1 / (4 * np.pi * np.pi)
        self.c = 5 / np.pi
        self.r = 6
        self.s = 10
        self.t = 1 / (8 * np.pi)
        
    def __call__(self, config):
        assert len(config) == self.d
        
        X = list(config.get_dictionary().values())
        x = (X[0] + 1) / 2 * 15 - 5
        y = (X[1] + 1) / 2 * 15
        res = self.a * np.power((y - self.b * x * x + self.c * x - self.r), 2) + self.s * (1 - self.t) * np.cos(x) + self.s
        
        return {'objective': res, 'context': None}

class Hartmann6(Test_Function):
    def __init__(self, dim_effective, dim_tot):
        super().__init__(dim_effective, dim_tot)
        
        self.alpha = np.array([1.0, 1.2, 3.0, 3.2])
        self.A = np.array([[10, 3, 17, 3.5, 1.7, 8],
                           [0.05, 10, 17, 0.1, 8, 14],
                           [3, 3.5, 1.7, 10, 17, 8],
                           [17, 8, 0.05, 10, 0.1, 14]])
        self.P = 1e-4 * np.array([[1312, 1696, 5569, 124, 8283, 5886],
                                  [2329, 4135, 8307, 3736, 1004, 9991],
                                  [2348, 1451, 3522, 2883, 3047, 6650],
                                  [4047, 8828, 8732, 5743, 1091, 381]])
    
    def __call__(self, config):
        assert len(config) == self.d
        
        X = list(config.get_dictionary().values())
        X = (np.array(X[:6]) + 1) / 2
        res = -np.dot(self.alpha, np.exp(np.sum(-self.A*np.power(X-self.P, 2), axis = -1)))
        
        return {'objective': res, 'context': None}

class Gramacy(Test_Function):
    def __init__(self, dim_effective, dim_tot):
        super().__init__(dim_effective, dim_tot)
        
    def __call__(self, config):
        assert len(config) == self.d
        
        X = list(config.get_dictionary().values())
        x = (X[0] + 1) / 2 * 8 - 2
        y = (X[1] + 1) / 2 * 8 - 2
        res = x * np.exp(- x * x - y * y)
        
        return {'objective': res, 'context': None}