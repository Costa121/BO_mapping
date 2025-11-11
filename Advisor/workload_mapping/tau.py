from .base import *
from openbox.surrogate.base.rf_with_instances import RandomForestWithInstances
from openbox.surrogate.base.build_gp import create_gp_model
from openbox.utils.util_funcs import get_types
import numpy as np
from sklearn.model_selection import KFold


class TAU(BaseMapper):
    def __init__(self, surrogate_type: str, seed: int = 0):
        super().__init__()

        self.method_id = 'tau'
        self.surrogate_type = surrogate_type
        self.surrogate_models = list()
        self.seed = seed

    def fit(self, source_hpo_data: list[History]):
        self.surrogate_models = list()
        config_space = source_hpo_data[0].config_space
        types, bounds = get_types(config_space)

        for history in source_hpo_data:

            surrogate = None
            if self.surrogate_type == 'prf':
                surrogate = RandomForestWithInstances(types=types, bounds=bounds, seed=self.seed)
            elif self.surrogate_type == 'gp':
                surrogate = create_gp_model(model_type='gp', config_space=config_space, types=types, bounds=bounds,
                                            rng=np.random.RandomState(self.seed))

            X = history.get_config_array()
            y = history.get_objectives()

            surrogate.train(X, y)
            self.surrogate_models.append(surrogate)

    @staticmethod
    def calculate_relative(A, B):
        P = 0
        l = len(A)
        for i in range(l):
            for j in range(i):
                if (A[i] <= A[j] and B[i] <= B[j]) or (A[i] > A[j] and B[i] > B[j]):
                    P += 1
        return 2 * P / (l * (l - 1))

    def map(self, target_history: History, source_hpo_data: list[History]) -> list[Tuple[int, float]]:

        X = target_history.get_config_array()
        y = target_history.get_objectives()

        types, bounds = get_types(target_history.config_space)
        surrogate = None
        if self.surrogate_type == 'prf':
            surrogate = RandomForestWithInstances(types=types, bounds=bounds, seed=self.seed)
        elif self.surrogate_type == 'gp':
            surrogate = create_gp_model(model_type='gp', config_space=target_history.config_space, types=types, bounds=bounds,
                                        rng=np.random.RandomState(self.seed))

        # 交叉验证10折验证
        k_fold_num = 10 if len(target_history) > 10 else len(target_history)
        kf = KFold(n_splits=k_fold_num)
        pred_y = np.zeros(len(y))
        for train_idx, val_idx in kf.split(X):
            X_train, X_val, y_train, y_val = X[train_idx, :], X[val_idx, :], y[train_idx], y[val_idx]
            surrogate.train(X_train, y_train)
            mu, _ = surrogate.predict(X_val)
            mu = mu.flatten()
            pred_y[val_idx] = mu

        tau = self.calculate_relative(pred_y, y)

        dists = [(-1, tau)]

        # 计算与每个源任务的相似度
        for i, surrogate_model in enumerate(self.surrogate_models):
            mu, _ = surrogate_model.predict(X)
            mu = mu.flatten()
            tau = self.calculate_relative(mu, y)
            dists.append((i, tau))

        dists = sorted(dists, key=lambda x: x[1], reverse=True)

        return dists