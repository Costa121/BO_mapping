from openbox.acquisition_function.acquisition import AbstractAcquisitionFunction
import numpy as np
from scipy.stats import norm

class piBO_acq(AbstractAcquisitionFunction):
    
    def __init__(self,
                 model,
                 prior,
                 beta = 5,
                 par: float = 0.0,
                 **kwargs):
        """Constructor

        Parameters
        ----------
        model : AbstractEPM
            A surrogate that implements at least
                 - predict_marginalized_over_instances(X)
        par : float, default=0.0
            Controls the balance between exploration and exploitation of the
            acquisition function.
        """

        super().__init__(model)
        
        self.long_name = 'piBO acquisition'
        self.beta = beta
        self.prior = prior
        self.par = par
        self.eta = None

    def _compute(self, X, **kwargs):
        
        if len(X.shape) == 1:
            X = X[:, np.newaxis]

        m, v = self.model.predict_marginalized_over_instances(X)
        s = np.sqrt(v)

        if self.eta is None:
            raise ValueError('No current best specified. Call update('
                             'eta=<int>) to inform the acquisition function '
                             'about the current best value.')

        def calculate_f():
            z = (self.eta - m - self.par) / s
            return (self.eta - m - self.par) * norm.cdf(z) + s * norm.pdf(z)

        if np.any(s == 0.0):
            s_copy = np.copy(s)
            s[s_copy == 0.0] = 1.0
            f = calculate_f()
            f[s_copy == 0.0] = 0.0
        else:
            f = calculate_f()
            
        if (f < 0).any():
            raise ValueError(
                "Expected Improvement is smaller than 0 for at least one "
                "sample.")
        
        prior_value = np.array([self.prior(X[i]) for i in range(X.shape[0])]).reshape(f.shape[0], 1)
        
        # f = f * np.power(prior_value, self.beta / self.num_data)
        f = f * np.power(prior_value, self.prior_weight)
        return f
    
class BOPro_acq(AbstractAcquisitionFunction):
    
    def __init__(self,
                 model,
                 prior,
                 beta = 5,
                 par: float = 0.0,
                 **kwargs):
        """Constructor

        Parameters
        ----------
        model : AbstractEPM
            A surrogate that implements at least
                 - predict_marginalized_over_instances(X)
        par : float, default=0.0
            Controls the balance between exploration and exploitation of the
            acquisition function.
        """

        super().__init__(model)
        
        self.long_name = 'piBO acquisition'
        self.beta = beta
        self.prior = prior
        self.par = par
        self.eta = None

    def _compute(self, X, **kwargs):
        
        if len(X.shape) == 1:
            X = X[:, np.newaxis]

        m, v = self.model.predict_marginalized_over_instances(X)
        s = np.sqrt(v)

        if self.eta is None:
            raise ValueError('No current best specified. Call update('
                             'eta=<int>) to inform the acquisition function '
                             'about the current best value.')

        def calculate_f():
            z = (self.eta - m - self.par) / s
            return norm.cdf(z)

        if np.any(s == 0.0):
            s_copy = np.copy(s)
            s[s_copy == 0.0] = 1.0
            f = calculate_f()
            f[s_copy == 0.0] = 0.0
        else:
            f = calculate_f()
            
        if (f < 0).any():
            raise ValueError(
                "Expected Improvement is smaller than 0 for at least one "
                "sample.")
        
        prior_value = np.array([self.prior(X[i]) for i in range(X.shape[0])]).reshape(f.shape[0], 1)
        good_prior = prior_value
        bad_prior = 1 - good_prior
        
        good_PI = f
        bad_PI = 1 - f
        
        # f = np.log(good_prior) - np.log(bad_prior) + self.num_data / self.beta * (np.log(good_PI) - np.log(bad_PI))
        f = self.prior_weight * (np.log(good_prior) - np.log(bad_prior)) + np.log(good_PI) - np.log(bad_PI)
        return f