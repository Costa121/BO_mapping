import numpy as np
from openbox.utils.constants import MAXINT
from openbox import logger

from openbox.utils.util_funcs import get_types
from .surrogate.tlbo.rgpe import RGPE
from .surrogate.tlbo.mfgpe import MFGPE
from .workload_mapping.distance import CosineMapper, EuclideanMapper
from .workload_mapping.rover import RoverMapper
import pandas as pd
from ConfigSpace import Configuration, ConfigurationSpace
from openbox.utils.history import Observation
from build_utils import SUCCESS, TIMEOUT, FAILED


def build_my_acq_func(func_str='ei', model=None, **kwargs):
    func_str = func_str.lower()
    if func_str.startswith('wrk'):
        inner_acq_func = func_str.split('_')[1]
        from .acquisition_function.weighted_rank import WeightedRank
        return WeightedRank(model=model, acq_func=inner_acq_func)
    elif func_str == 'ei':
        from openbox.acquisition_function import EI
        return EI(model=model)
    elif func_str == 'pibo':
        from .acquisition_function.expert_acq import piBO_acq
        return piBO_acq(model=model, **kwargs)
    elif func_str == 'bo_pro':
        from .acquisition_function.expert_acq import BOPro_acq
        return BOPro_acq(model=model, **kwargs)
    else:
        raise ValueError('Invalid string %s for acquisition function!' % func_str)


def build_my_compressor(X, y, k=23, func_str='none'):
    from shap import Explainer
    def _shap_importance(X, model):
        explainer = Explainer(model)
        shap_values = np.abs(explainer(X).values).mean(axis=0)
        return shap_values

    importances = []
    models = []
    for i in range(len(X)):
        tX = X[i]
        ty = y[i]
        if func_str in ['ottertune', 'rover-l', 'rover', 'rover-s']:
            from xgboost import XGBRegressor
            model = XGBRegressor().fit(tX, ty)
            models.append(model)
            importances.append(_shap_importance(tX, model))
        elif func_str == 'rover-g':
            from sklearn.ensemble import RandomForestRegressor
            model = RandomForestRegressor().fit(tX, ty)
            models.append(model)
            importances.append(model.feature_importances_)
        elif func_str == 'opadviser':
            from lightgbm import LGBMRegressor
            model = LGBMRegressor().fit(tX, ty)   # LGBM 不需要特征选择
            return model, list(range(tX.shape[1]))
        else:
            raise ValueError(f"Unknown func_str: {func_str}!")

    importances = np.mean(np.array(importances), axis = 0)
    k = min(k, len(importances))
    return models, sorted(np.argsort(-importances).tolist())


def build_my_surrogate(func_str='gp', config_space=None, rng=None, transfer_learning_history=None, **kwargs):
    extra_dim = kwargs.get('extra_dim', 0)
    seed = kwargs.get('seed', 42)
    norm_y = kwargs.get('norm_y', True)
    indices = kwargs.get('indices', None)

    assert config_space is not None
    func_str = func_str.lower()
    types, bounds = get_types(config_space)
    if extra_dim > 0:
        types = np.hstack((types, np.zeros(extra_dim, dtype=np.uint)))
        bounds = np.vstack((bounds, np.array([[0, 1]] * extra_dim)))

    if indices and len(indices) == 2 and indices[1]:
        selected, fixed = indices
        types = np.concatenate((types[selected], types[fixed]))
        bounds = np.concatenate((bounds[selected], bounds[fixed]))
        # logger.warn("Types and bounds: %s and %s" % (types, bounds))

    if func_str == 'prf':
        try:
            from openbox.surrogate.base.rf_with_instances import RandomForestWithInstances
            return RandomForestWithInstances(types=types, bounds=bounds, seed=seed)
        except ModuleNotFoundError:
            from openbox.surrogate.base.rf_with_instances_sklearn import skRandomForestWithInstances
            logger.warning('[Build Surrogate] Use probabilistic random forest based on scikit-learn. '
                           'For better performance, please install pyrfr: '
                           'https://open-box.readthedocs.io/en/latest/installation/install_pyrfr.html')
            return skRandomForestWithInstances(types=types, bounds=bounds, seed=seed)

    elif func_str.startswith('gp'):
        from openbox.surrogate.base.build_gp import create_gp_model
        return create_gp_model(model_type=func_str[:2],
                               config_space=config_space,
                               types=types,
                               bounds=bounds,
                               rng=rng)
    elif func_str.startswith('mce'):
        inner_model = func_str.split('_')[1]
        return RGPE(config_space=config_space, source_hpo_data=transfer_learning_history, seed=seed,
                    surrogate_type=inner_model, norm_y=norm_y)
    elif func_str.startswith('re'):
        inner_model = func_str.split('_')[1]
        return MFGPE(config_space=config_space, source_hpo_data=transfer_learning_history, seed=seed,
                     surrogate_type=inner_model, norm_y=norm_y)
    elif func_str.startswith('mfse'):
        inner_model = func_str.split('_')[1]
        return MFGPE(config_space=config_space, source_hpo_data=None, seed=seed,
                     surrogate_type=inner_model, norm_y=norm_y)

    else:
        raise ValueError('Invalid string %s for surrogate!' % func_str)


def calculate_ranking(score_list, ascending=False):
    rank_list = list()
    for i in range(len(score_list)):
        value_list = pd.Series(list(score_list[i]))
        rank_array = np.array(value_list.rank(ascending=ascending))
        rank_list.append(rank_array)

    return rank_list


def map_source_hpo_data(map_strategy, target_his, source_hpo_data, **kwargs):
    sims = None
    if 'cos' in map_strategy:
        mapper = CosineMapper()
        sims = mapper.map(target_his, source_hpo_data)
    elif 'euc' in map_strategy:
        mapper = EuclideanMapper()
        sims = mapper.map(target_his, source_hpo_data)
    elif 'rover' in map_strategy:
        inner_sm = kwargs.get('inner_surrogate_model', 'gp')
        rover = RoverMapper(surrogate_type=inner_sm)
        rover.fit(source_hpo_data)
        sims = rover.map(target_his, source_hpo_data)
    else:
        raise ValueError('Invalid ws_strategy: %s' % map_strategy)

    return sims


def build_observation(config, results, **kwargs):

    last_context = kwargs.get('last_context', None)

    ret, timeout_status, traceback_msg, elapsed_time = (
        results['result'], results['timeout'], results['traceback'], results['elapsed_time'])

    perf, context = ret['objective'], ret['context']

    if timeout_status:
        trial_state = TIMEOUT
    elif traceback_msg is not None:
        trial_state = FAILED
        logger.error(f'Exception in objective function:\n{traceback_msg}\nconfig: {config}')
    else:
        trial_state = SUCCESS

    obs = Observation(config=config, objectives=[perf], trial_state=trial_state, elapsed_time=elapsed_time,
                      extra_info={'last_context': last_context, 'context': context, 'origin': config.origin})

    return obs

def transfer_config_to_origin_space(config, origin_space, A):
    config_dict = config.get_dictionary()
    config_array = np.array(list(config_dict.values()))
    new_config = A @ config_array
    new_config[new_config < -1] = -1
    new_config[new_config > 1] = 1
    new_dict = dict()
    for i, param in enumerate(origin_space.get_hyperparameters()):
        new_dict[param.name] = new_config[i]
    new_config = Configuration(configuration_space=origin_space, values=new_dict)

    return new_config