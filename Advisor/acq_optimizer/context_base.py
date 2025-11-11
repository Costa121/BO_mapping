import numpy as np
import pickle as pkl
import math

from openbox.utils.config_space.util import convert_configurations_to_array


class AcquisitionOptimizer:
    def __init__(self, acquisition_function, config_space, rng,
                 context, context_flag=False):
        self.acq = acquisition_function
        self.config_space = config_space
        if rng is None:
            self.rng = np.random.RandomState(seed=42)
        else:
            self.rng = rng
        self.iter_id = 0
        self.context = context
        self.context_flag = context_flag


    def maximize(self, observations, num_points, **kwargs):
        return [t[1] for t in self._maximize(observations, num_points, **kwargs)]

    def _maximize(self, observations, num_points: int, **kwargs):
        raise NotImplementedError()

    def _sort_configs_by_acq_value(self, configs, **kwargs):
        acq_values = self.acquisition_function(configs, **kwargs)
        random = self.rng.rand(len(acq_values))
        # Last column is primary sort key!
        indices = np.lexsort((random.flatten(), acq_values.flatten()))
        # Cannot use zip here because the indices array cannot index the
        # rand_configs list, because the second is a pure python list
        return [(acq_values[ind][0], configs[ind]) for ind in indices[::-1]]

    def acquisition_function(self, configs, **kwargs):
        X = convert_configurations_to_array(configs)
        if self.context_flag:
            _context = np.tile(self.context, (X.shape[0], 1))
            X = np.hstack((X, _context))

        return self.acq(X, convert=False, **kwargs)