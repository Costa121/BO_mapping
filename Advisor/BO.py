from .base import BaseAdvisor
from .acq_optimizer.context_local_random import InterleavedLocalAndRandomSearch

import numpy as np
import math
import copy
from itertools import combinations
from sklearn.random_projection import GaussianRandomProjection
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from scipy.optimize import minimize
from ConfigSpace import Configuration, ConfigurationSpace, UniformFloatHyperparameter
from openbox import logger
from openbox import space as sp
from openbox.utils.config_space.util import convert_configurations_to_array
from openbox.utils.history import Observation, History
from .rover.transfer import get_transfer_suggestion
from .utils import build_my_surrogate, build_my_acq_func, build_my_compressor, transfer_config_to_origin_space, build_observation

from datetime import datetime

class BO(BaseAdvisor):
    def __init__(self, config_space: ConfigurationSpace, source_hpo_data=None,
                 surrogate_type='prf', acq_type='ei', task_id='test', meta_feature=None,
                 ws_strategy='none', ws_args={'init_num': 5}, tl_args={'topk': 5},
                 ep_args=None, ep_strategy='none',
                 cprs_k=50, space_history = None, cprs_strategy='none',
                 safe_flag=False, seed=42, rng=None, rand_prob=0.15, rand_mode='ran', **kwargs):
        super().__init__(config_space, source_hpo_data=source_hpo_data, task_id=task_id,
                         ws_strategy=ws_strategy, ws_args=ws_args, tl_args=tl_args, 
                         ep_args=ep_args, ep_strategy=ep_strategy, meta_feature=meta_feature,
                         cprs_strategy=cprs_strategy, space_history=space_history, cprs_k=cprs_k,
                         seed=seed, rng=rng, rand_prob=rand_prob, rand_mode=rand_mode, **kwargs)
    
        self.safe_flag = safe_flag

        self.acq_type = acq_type
        self.surrogate_type = surrogate_type
        self.extra_dim = 0
        
        self.fixed_indices = []
        self.current_indices = list(range(len(self.origin_config_space)))

        self.norm_y = True
        if 'wrk' in acq_type:
            self.norm_y = False

        self.init_num = ws_args['init_num']
        
        if self.cprs_strategy == 'REMBO':
            self.A = self.rng.randn(len(self.origin_config_space), self.cprs_k)
            self.meta_feature['A'] = self.A.tolist()
            self.config_space = ConfigurationSpace()
            name_len = len(str(self.cprs_k))
            for i in range(self.cprs_k):
                i_str = str(i)
                name = 'x' + '0' * (name_len - len(i_str)) + i_str
                self.config_space.add_hyperparameter(sp.Real(name, lower=-np.sqrt(self.cprs_k), upper=np.sqrt(self.cprs_k), default_value=0))

         # 初始化 ALEBO 相关参数
        if self.cprs_strategy == 'ALEBO':
            self.D = len(self.origin_config_space)
            self.d = self.cprs_k
            random_matrix = self.rng.randn(self.D, self.D)
            Q_full, R = np.linalg.qr(random_matrix)
            self.B = Q_full[:self.d, :].copy()
            self.Binv = self.B.T
            self.Q = self.Binv @ self.B
            self.meta_feature['B'] = self.B.tolist()
            self.meta_feature['Binv'] = self.Binv.tolist()
            self.meta_feature['Q'] = self.Q.tolist()
            self.meta_feature['alebo_d'] = self.d
            self.meta_feature['alebo_D'] = self.D
            self.config_space = ConfigurationSpace()
            name_len = len(str(self.cprs_k))
            bound = 100.0
            for i in range(self.cprs_k):
                i_str = str(i)
                name = 'x' + '0' * (name_len - len(i_str)) + i_str
                self.config_space.add_hyperparameter(
                    sp.Real(name, lower=-bound, upper=bound, default_value=0.0)
                )
            # ALEBO特有参数
            self.alebo_b = 2.0  # 初始化采样的边界参数
            self.alebo_max_attempts = 1000  # 拒绝采样最大尝试次数
            self.alebo_projection_tolerance = 1e-5  # 投影容差

        if self.cprs_strategy == 'BAxUS':
            # === BAxUS 核心参数 ===
            self.baxus_d_init = kwargs.get('baxus_d_init', 5)  # _init_target_dim
            self.baxus_n_init = kwargs.get('baxus_n_init', 10)  # n_init
            self.baxus_length_init = kwargs.get('baxus_length_init', 0.8)  # length_init
            self.baxus_length_min = kwargs.get('baxus_length_min', 0.5**7)  # length_min
            self.baxus_length_max = kwargs.get('baxus_length_max', 1.6)  # length_max
            self.baxus_succtol = kwargs.get('baxus_succtol', 3)  # succtol
            self.baxus_n_new_bins = kwargs.get('baxus_n_new_bins', 1)  # behavior.n_new_bins

            # === BAxUS 状态变量 ===
            self.origin_config_space = config_space
            self.input_dim = len(self.origin_config_space.get_hyperparameters())  # _input_dim
            
            self.target_dim = self.baxus_d_init  # target_dim
            self._init_target_dim = self.baxus_d_init
            self._target_dim_after_reset = self.target_dim
            
            # 信任区域状态
            self.length = self.baxus_length_init  # 当前信任区域长度
            self.failcount = 0  # failcount
            self.succcount = 0  # succcount
            
            # BAxUS 特有状态
            self._axus_change_iterations = []  # 维度变化的迭代点
            self._split_points = []  # 分割点
            self._trust_region_restarts = []  # 信任区域重启点
            self._dim_in_iterations = {}  # 各迭代的维度记录
            
            # 投影相关
            self._baxus_create_projection()
            
            # TR 内部数据
            self._X_tr = np.empty((0, self.target_dim))  # TR内的低维数据
            self._fX_tr = np.empty((0, 1))  # TR内的目标值
            
            print(f"[BAxUS] Initialized: input_dim={self.input_dim}, target_dim={self.target_dim}")

        # Add-GP-UCB 相关参数初始化
        if self.cprs_strategy == 'Add-GP-UCB':
            # 使用默认值，这样就不需要修改build_utils.py
            self.addgp_d_max = kwargs.get('addgp_d_max', min(5, len(self.origin_config_space)))
            self.addgp_sigma_sms = kwargs.get('addgp_sigma_sms', 1.0)
            self.addgp_sigma_prs = kwargs.get('addgp_sigma_prs', 1.0)
            self.addgp_common_noise = kwargs.get('addgp_common_noise', 0.01)
            self.addgp_beta_coeff = kwargs.get('addgp_beta_coeff', 2.0)
            
            # 创建维度分解
            self._create_addgp_decomposition()
            logger.info(f"[Add-GP-UCB] Initialized with default parameters")
            
        # RPP-GP-UCB 相关参数初始化
        if self.cprs_strategy == 'RPP-GP-UCB':
            self.rpp_d_max = kwargs.get('rpp_d_max', min(3, len(self.origin_config_space)))
            self.rpp_M = kwargs.get('rpp_M', min(5, max(1, len(self.origin_config_space) // self.rpp_d_max)))
            self.rpp_delta = kwargs.get('rpp_delta', 0.2)  # δ-ratio约束参数
            self.rpp_N_cyc = kwargs.get('rpp_N_cyc', 10)  # 投影矩阵更新频率
            self.rpp_sigma_sms = kwargs.get('rpp_sigma_sms', 1.0)
            self.rpp_sigma_prs = kwargs.get('rpp_sigma_prs', 1.0)
            self.rpp_common_noise = kwargs.get('rpp_common_noise', 0.01)
            self.rpp_beta_coeff = kwargs.get('rpp_beta_coeff', 2.0)
            
            # RPP-GP-UCB状态变量
            self.rpp_W_hat_list = []  # 带约束的投影矩阵列表
            self.rpp_gp_models = []   # 每个子空间的GP模型
            self.rpp_prev_W = None    # warm start用的上一轮投影矩阵
            self.rpp_common_mean = 0.0
            
            logger.info(f"[RPP-GP-UCB] Initialized with d_max={self.rpp_d_max}, M={self.rpp_M}, delta={self.rpp_delta}")
        
        self._reset_config_space()
        # History 里面存的config是完整的原始空间配置
        # 取 observation 出来的时候需要根据 indices 进行变换得到压缩后的配置
        # Challenger 得到的配置则是压缩后的，需要再把它拼回完整的
        self.ini_configs = list()
        self.default_config = self.config_space.get_default_configuration()
        self.default_config.origin = "default"
        self.ini_configs.append(self.default_config)

    def warm_start(self):
        if self.ws_strategy == 'none':
            return

        sims = self.source_hpo_data_sims

        for i, sim in enumerate(sims):
            logger.info("The %d-th similar task(%s): %s" % (sim[0], self.source_hpo_data[i].task_id, sim[1]))

        warm_str_list = []
        for i in range(len(sims)):
            idx, sim = sims[i]
            task_str = self.source_hpo_data[idx].task_id
            warm_str = "%s: sim%.4f" % (task_str, sim)
            warm_str_list.append(warm_str)

        if 'warm_start' not in self.history.meta_info:
            self.history.meta_info['warm_start'] = [warm_str_list]
        else:
            self.history.meta_info['warm_start'].append(warm_str_list)

        num_evaluated = len(self.history)
        if self.ws_strategy.startswith('best'):
            for i, sim in enumerate(sims):
                sim_obs = copy.deepcopy(self.source_hpo_data[sim[0]].observations)
                sim_obs = sorted(sim_obs, key=lambda x: x.objectives[0])

                task_num = 3 if i == 0 else 1
                for j in range(task_num):
                    config_warm = sim_obs[j].config
                    config_warm.origin = self.ws_strategy + self.source_hpo_data[sim[0]].task_id
                    # 后加的更差，因为是从后往前取的，所以往前加
                    self.ini_configs = [config_warm] + self.ini_configs
                if len(self.ini_configs) + num_evaluated >= self.init_num:
                    break

            while len(self.ini_configs) + num_evaluated < self.init_num:
                config = self.sample_random_configs(self.config_space, 1,
                                                    excluded_configs=self.history.configurations)[0]
                self.ini_configs = [config] + self.ini_configs

            logger.info("Successfully warm start %d configurations with %s!" % (len(self.ini_configs), self.ws_strategy))

        elif self.ws_strategy.startswith('rgpe'):
            topk = self.ws_args.get('topk', 3)

            src_history = [self.source_hpo_data[sims[i][0]] for i in range(topk)]
            target_history = self.history

            while len(self.ini_configs) + num_evaluated < self.init_num:
                final_config = get_transfer_suggestion(src_history, target_history, _logger_kwargs=self._logger_kwargs)
                final_config.origin = self.ws_strategy
                if final_config not in self.history.configurations + self.ini_configs:
                    self.ini_configs.append(final_config)

            logger.info("Successfully warm start %d configurations with %s!" % (len(self.ini_configs), self.ws_strategy))

        else:

            raise ValueError('Invalid ws_strategy: %s' % self.ws_strategy)

    """
    采样(使用ini_configs进行热启动和普通采样)
    以及安全约束 (40轮后阈值为 0.85 * incumbent_value)
    """
    def sample(self, return_list=False):
        num_config_evaluated = len(self.history)
        if len(self.ini_configs) == 0 and num_config_evaluated < self.init_num:
            self.warm_start()

        if num_config_evaluated < self.init_num:
            if len(self.ini_configs) > 0:
                config = self.ini_configs[-1]
                self.ini_configs.pop()
            else:
                config = self.sample_random_configs(self.config_space, 1,
                                                    excluded_configs=self.history.configurations)[0]
            if return_list:
                return [config]
            else:
                # 统一处理配置转换
                if self.cprs_strategy == 'REMBO':
                    self.config_in_new_space = config
                    config = transfer_config_to_origin_space(config, self.origin_config_space, self.A)
                elif self.cprs_strategy == 'ALEBO':
                    self.config_in_new_space = config
                    config = self._transform_config(config)
                elif self.cprs_strategy == 'Add-GP-UCB':
                    # Add-GP-UCB在初始化阶段直接使用原始空间
                    pass
                return config

        # Add-GP-UCB使用特殊的采样逻辑
        if self.cprs_strategy == 'Add-GP-UCB':
            return self._sample_addgp_ucb(return_list)
        
        # BAxUS使用特殊的采样逻辑
        if self.cprs_strategy == 'BAxUS':
            return self._sample_baxus(return_list)
        
        # Gr-VS使用特殊的采样逻辑
        if self.cprs_strategy == 'Gr-VS':
            return self._sample_grvs()
        
        # Dropout-VS进行随机调优维度选取
        if self.cprs_strategy == 'Dropout-VS':
            indices = sorted(self.rng.choice(np.arange(len(self.origin_config_space)), size = self.cprs_k, replace = False).tolist())
            self._set_space_by_indices(indices)
            
        # 我们的方法暂时采用SHAP值进行筛选
        if self.cprs_strategy == 'ours':
            from shap import Explainer
            from xgboost import XGBRegressor
            tX = self.history.get_config_array()
            ty = self.history.get_objectives()
            xgb = XGBRegressor().fit(tX, ty)
            explainer = Explainer(xgb)
            shap_values = np.abs(explainer(tX).values).mean(axis=0)
            indices = sorted(np.argsort(-shap_values)[:self.cprs_k].tolist())
            self._set_space_by_indices(indices)
        
        # 获取历史数据和构建observations
        if self.cprs_strategy == 'ALEBO':
            original_configs = self.history.configurations
            compressed_configs = []
            X = []
            
            for orig_config in original_configs:
                compressed_config = self._inverse_transform_config(orig_config)
                compressed_configs.append(compressed_config)
                compressed_array = convert_configurations_to_array([compressed_config])[0]
                X.append(compressed_array)
            
            X = np.array(X) if X else np.empty((0, self.d))
            
            # 创建修改后的observations
            observations = []
            for i, compressed_config in enumerate(compressed_configs):
                # 复制原始observation并替换config
                orig_obs = self.history.observations[i]
                # 创建新的observation对象，保持其他属性不变
                modified_obs = type(orig_obs)(
                    config=compressed_config,
                    objectives=orig_obs.objectives,
                    constraints=getattr(orig_obs, 'constraints', None),
                    trial_state=orig_obs.trial_state,
                    elapsed_time=orig_obs.elapsed_time
                )
                observations.append(modified_obs)
        else:
            X = self.history.get_config_array()
            if self.fixed_indices:
                X = X[:, self.current_indices]
                
            observations = self.history.observations
        
        Y = self.history.get_objectives()

        if self.surrogate_type == 'gpf':
            self.surrogate = build_my_surrogate(func_str=self.surrogate_type, config_space=self.config_space,
                                                rng=self.rng,
                                                transfer_learning_history=self.source_hpo_data,
                                                extra_dim=self.extra_dim, norm_y=self.norm_y)
            logger.info("Successfully rebuild the surrogate model GP!")
            
        self.surrogate.train(X, Y)

        incumbent_value = self.history.get_incumbent_value()
        self.acq_func.update(model=self.surrogate,
                            eta=incumbent_value,
                            num_data=num_config_evaluated)

        if self.fixed_indices:
            tmp_obs = []
            for obs in self.history.observations:
                new_obs = copy.deepcopy(obs)
                new_obs.config = Configuration(configuration_space=self.config_space,
                                               values={name: obs.config[name] for name in self.param_names})
                tmp_obs.append(new_obs)
            observations = tmp_obs

        challengers = self.acq_optimizer.maximize(observations=observations, num_points=2000)

        if return_list:
            return challengers.challengers

        cur_config = challengers.challengers[0]
        
        # 安全约束逻辑保持不变
        if self.safe_flag:
            recommend_flag = True
            if len(self.history) >= 10:
                recommend_flag = False
                for config in challengers.challengers:
                    X = convert_configurations_to_array([config])
                    if self.fixed_indices:
                        X = np.hstack((X[:, self.current_indices], X[:, self.fixed_indices]))
                    pred_mean, _ = self.surrogate.predict(X)
                    if pred_mean[0] < 0.85 * self.history.get_incumbent_value():
                        logger.warn(
                            '-----------The config_%d meet the security constraint-----------' % challengers.challengers.index(
                                config))
                        cur_config = config
                        recommend_flag = True
                        break

            if recommend_flag:
                logger.warn("Successfully recommend a configuration through Advisor!")
            else:
                logger.error(
                    "Failed to recommend a configuration that meets the security constraint! Return the incumbent_config")
                cur_config = self.history.get_incumbent_configs()[0]
        else:
            recommend_flag = False
            for config in challengers.challengers:
                incumbent_config = self.history.get_incumbent_configs()[0]
                
                if self.fixed_indices:
                    ori_param_names = [param.name for param in self.origin_config_space.get_hyperparameters()]
                    ret_conf = {
                        name: config[name] if idx in self.current_indices else incumbent_config[name]
                        for idx, name in enumerate(ori_param_names)
                    }
                    config = Configuration(configuration_space=self.origin_config_space, values=ret_conf)
                
                # 统一的最终配置转换处理
                if self.cprs_strategy == 'REMBO':
                    self.config_in_new_space = config
                    config = transfer_config_to_origin_space(config, self.origin_config_space, self.A)
                elif self.cprs_strategy == 'ALEBO':
                    config = self._transform_config(config)
                    
                if config not in self.history.configurations:
                    cur_config = config
                    recommend_flag = True
                    break

            if recommend_flag:
                logger.warn("Successfully recommend a configuration through Advisor!")
            else:
                logger.error("Failed to recommend am unique configuration ! Return a random config")
                cur_config = self.sample_random_configs(self.origin_config_space, 1, excluded_configs=self.history.configurations)[0]

        return cur_config
    
    '''
    每次进行空间重压缩的时候调用，更新优化器
    '''
    def _set_space_by_indices(self, indices):
        if not indices:  # 如果 indices 为空或 None，直接使用原空间
            logger.info(f"Using original ConfigurationSpace (no compression).")
            self.config_space = copy.deepcopy(self.origin_config_space)
            self.fixed_indices = []
            self.current_indices = list(range(len(self.config_space)))
        else:
            logger.info(f"Use algorithm [{self.cprs_strategy}] to reset search space, only use indices: {indices}")
            self.current_indices = indices
            self.fixed_indices = [i for i in range(len(self.origin_config_space)) if i not in indices]
            self.config_space = ConfigurationSpace()
            for idx in indices:
                name = self.origin_config_space.get_hyperparameter_by_idx(idx)
                if name in self.origin_config_space:
                    self.config_space.add_hyperparameter(self.origin_config_space[name])

        self.surrogate = build_my_surrogate(func_str=self.surrogate_type, config_space=self.config_space, rng=self.rng,
                                            transfer_learning_history=self._source_hpo_data_in_new_space(),
                                            extra_dim=self.extra_dim, norm_y=self.norm_y)
        
        _kwargs = {}
        self.acq_func = build_my_acq_func(func_str=self.acq_type, model=self.surrogate, **_kwargs)

    
        self.acq_optimizer = InterleavedLocalAndRandomSearch(acquisition_function=self.acq_func,
                                                             rand_prob=self.rand_prob, rand_mode=self.rand_mode,
                                                             config_space=self.config_space, rng=self.rng,
                                                             context=None, context_flag=False)
        self.config_space_seed = self.rng.randint(2 ** 31 - 1)
        self.config_space.seed(self.config_space_seed)
        self.param_names = [param.name for param in self.config_space.get_hyperparameters()]
        logger.info("Compressed configuration space: %s" % (self.param_names))

    def _source_hpo_data_in_new_space(self):
            if self.source_hpo_data is None:
                return None
            
            new_his = []
            
            param_names = [param.name for param in self.config_space.get_hyperparameters()]
            for his in self.source_hpo_data:
                data = {
                    'task_id': his.task_id,
                    'num_objectives': his.num_objectives,
                    'num_constraints': his.num_constraints,
                    'ref_point': his.ref_point,
                    'meta_info': his.meta_info,
                    'global_start_time': his.global_start_time.isoformat(),
                    'observations': [
                        obs.to_dict() for obs in his.observations
                    ]
                }

                for obs in data['observations']:
                    new_conf = {}
                    for name in param_names:
                        new_conf[name] = obs['config'][name]
                    obs['config'] = new_conf
                    
                global_start_time = data.pop('global_start_time')
                global_start_time = datetime.fromisoformat(global_start_time)
                observations = data.pop('observations')
                observations = [Observation.from_dict(obs, self.config_space) for obs in observations]

                history = History(**data)
                history.global_start_time = global_start_time
                history.update_observations(observations)
                
                new_his.append(history)
            
            return new_his
        
    def update(self, config, results):
        if self.cprs_strategy == 'REMBO':
            obs = build_observation(config, results, last_context=self.last_context)
            self.origin_history.update_observation(obs)
            obs = build_observation(self.config_in_new_space, results, last_context=self.last_context)
            self.history.update_observation(obs)
        elif self.cprs_strategy == 'ALEBO':
            obs = build_observation(config, results, last_context=self.last_context)
            self.history.update_observation(obs)
        elif self.cprs_strategy == 'BAxUS':
            # 先更新历史
            obs = build_observation(config, results, last_context=self.last_context)
            self.history.update_observation(obs)
            
            # 获取性能值并调整TR长度
            perf_value = None
            if 'perf' in results:
                perf_value = results['perf']
            elif 'result' in results and 'objs' in results['result']:
                perf_value = results['result']['objs'][0]
            elif obs.objectives and len(obs.objectives) > 0:
                perf_value = obs.objectives[0]
            
            if perf_value is not None:
                self._adjust_length(perf_value)  # 修复：使用正确的方法名
            else:
                print("[BAxUS] Warning: Could not extract performance value for TR update")
        else:
            obs = build_observation(config, results, last_context=self.last_context)
            self.history.update_observation(obs)
        
        if 'result' in results and 'context' in results['result']:
            self.last_context = results['result']['context']

    def _reset_config_space(self):
        self.history = History(task_id=self.task_id, config_space=self.config_space, meta_info=self.meta_feature)
        
        self.surrogate = build_my_surrogate(func_str=self.surrogate_type, config_space=self.config_space, rng=self.rng,
                                            transfer_learning_history=self._source_hpo_data_in_new_space(),
                                            extra_dim=self.extra_dim, norm_y=self.norm_y)
        
        _kwargs = {}
        self.acq_func = build_my_acq_func(func_str=self.acq_type, model=self.surrogate, **_kwargs)
        
        self.acq_optimizer = InterleavedLocalAndRandomSearch(acquisition_function=self.acq_func,
                                                            rand_prob=self.rand_prob, rand_mode=self.rand_mode,
                                                            config_space=self.config_space, rng=self.rng,
                                                            context=None, context_flag=False)

        self.config_space.seed(self.config_space_seed)
        self.param_names = [param.name for param in self.config_space.get_hyperparameters()]

     # =================== ALEBO核心方法 ===================

    def _alebo_sample_initial_points(self, n_points, b=None):
        """
        ALEBO初始化采样器：使用拒绝采样在有效子空间内生成点
        
        参数:
            n_points: 需要的点数
            b: 采样边界，默认使用self.alebo_b
        
        返回:
            numpy.ndarray: shape (n_points, d) 的低维点
        """
        if b is None:
            b = self.alebo_b
        
        print(f"[ALEBO] 开始初始化采样，目标点数={n_points}, 边界=±{b}")
        
        valid_points = []
        total_attempts = 0
        batch_size = max(50, n_points * 5)  # 批量采样提高效率
        
        while len(valid_points) < n_points and total_attempts < self.alebo_max_attempts:
            # 1. 在[-b, b]^d空间中生成均匀随机点
            X_low = self.rng.uniform(-b, b, size=(batch_size, self.d))
            
            # 2. 通过Q矩阵投影到B定义的子空间
            # X_projected = X_low @ Q^T，然后投影回高维空间
            X_subspace = X_low @ self.Q.T  # 投影到子空间并扩展到高维
            
            # 3. 检查投影后的点是否在[-1, 1]^D范围内
            # 这是ALEBO的关键约束：确保高维点在可行域内
            in_bounds = np.all(np.abs(X_subspace) <= 1.0, axis=1)
            valid_indices = np.where(in_bounds)[0]
            
            # 4. 收集有效的低维点
            for idx in valid_indices:
                if len(valid_points) < n_points:
                    valid_points.append(X_low[idx])
                else:
                    break
            
            total_attempts += batch_size
            
            if total_attempts % (batch_size * 5) == 0:
                print(f"[ALEBO] 采样进度: {len(valid_points)}/{n_points} "
                    f"(尝试={total_attempts}, 成功率={len(valid_points)/total_attempts*100:.1f}%)")
        
        # 如果拒绝采样失败，使用fallback策略
        if len(valid_points) < n_points:
            print(f"[ALEBO] 警告：拒绝采样只获得了{len(valid_points)}个有效点")
            
            # 使用较小的边界重试
            if b > 0.5:
                print(f"[ALEBO] 使用较小边界重试: {b/2}")
                additional_points = self._alebo_sample_initial_points(
                    n_points - len(valid_points), b=b/2
                )
                valid_points.extend(additional_points)
            else:
                # 最后手段：直接在小范围内采样
                remaining = n_points - len(valid_points)
                fallback_points = self.rng.uniform(-0.5, 0.5, size=(remaining, self.d))
                valid_points.extend(fallback_points)
                print(f"[ALEBO] 使用fallback策略生成剩余{remaining}个点")
        
        result = np.array(valid_points[:n_points])
        print(f"[ALEBO] 初始化采样完成，生成{len(result)}个点")
        return result

    def _alebo_project_to_high_dim(self, X_low):
        """
        将低维点投影到高维空间
        
        参数:
            X_low: shape (n, d) 的低维点
        
        返回:
            numpy.ndarray: shape (n, D) 的高维点
        """
        if X_low.ndim == 1:
            X_low = X_low.reshape(1, -1)
        
        # 使用伪逆矩阵投影：X_high = X_low @ Binv^T
        X_high = X_low @ self.Binv.T
        
        # 裁剪到[-1, 1]^D确保在可行域内
        X_high_clipped = np.clip(X_high, -1.0, 1.0)
        
        # 检查裁剪的影响
        clipping_effect = np.abs(X_high - X_high_clipped).max()
        if clipping_effect > self.alebo_projection_tolerance:
            print(f"[ALEBO] 警告：投影裁剪影响较大，最大偏差={clipping_effect:.2e}")
        
        return X_high_clipped

    def _alebo_project_to_low_dim(self, X_high):
        """
        将高维点投影到低维空间
        
        参数:
            X_high: shape (n, D) 的高维点或 (D,) 的单个高维点
        
        返回:
            numpy.ndarray: shape (n, d) 的低维点或 (d,) 的单个低维点
        """
        if X_high.ndim == 1:
            X_high = X_high.reshape(1, -1)
            return_1d = True
        else:
            return_1d = False
        
        # 使用投影矩阵B：X_low = X_high @ B^T
        X_low = X_high @ self.B.T
        
        # 如果输入是单个点，返回一维数组
        if return_1d:
            X_low = X_low.flatten()
        
        return X_low

    def _alebo_validate_projection(self, X_low):
        """
        验证低维点通过投影后是否仍在有效域内
        
        参数:
            X_low: shape (n, d) 的低维点
        
        返回:
            tuple: (is_valid, X_high, reconstruction_error)
        """
        # 投影到高维
        X_high = self._alebo_project_to_high_dim(X_low)
        
        # 检查是否在[-1,1]^D内
        is_valid = np.all(np.abs(X_high) <= 1.0 + self.alebo_projection_tolerance, axis=1)
        
        # 计算重构误差
        X_low_reconstructed = self._alebo_project_to_low_dim(X_high)
        reconstruction_error = np.linalg.norm(X_low - X_low_reconstructed, axis=1)
        
        return is_valid, X_high, reconstruction_error

    def _transform_config(self, config):
            if self.cprs_strategy == 'ALEBO':
                # 1. 提取低维数组
                if isinstance(config, Configuration):
                    config_array = convert_configurations_to_array([config])[0]
                else:
                    config_array = np.array(config)
    
                # 2. 投影到高维空间
                origin_array = self._alebo_project_to_high_dim(config_array)
                # 确保 origin_array 是一维数组
                origin_array = origin_array.flatten()
                
                # 3. 验证投影有效性
                is_valid, _, recon_error = self._alebo_validate_projection(config_array)
                if not is_valid[0]:
                    print(f"[ALEBO] 警告：投影点不在有效域内")
                if recon_error[0] > self.alebo_projection_tolerance:
                    print(f"[ALEBO] 警告：重构误差较大 = {recon_error[0]:.2e}")
                
                # 4. 转换为原始配置空间的配置
                origin_config_dict = {}
                hp_names = self.origin_config_space.get_hyperparameter_names()
                
                for i, hp_name in enumerate(hp_names):
                    hp = self.origin_config_space.get_hyperparameter(hp_name)
                    normalized_value = origin_array[i]  # 已经在[-1,1]范围内
                    
                    if isinstance(hp, sp.Real):
                        # 线性映射到实际范围
                        actual_value = (normalized_value + 1) / 2 * (hp.upper - hp.lower) + hp.lower
                        # 确保在边界内
                        actual_value = np.clip(actual_value, hp.lower, hp.upper)
                        origin_config_dict[hp_name] = float(actual_value)
                        
                    elif isinstance(hp, sp.Integer):
                        # 线性映射并四舍五入
                        actual_value = (normalized_value + 1) / 2 * (hp.upper - hp.lower) + hp.lower
                        actual_value = int(round(np.clip(actual_value, hp.lower, hp.upper)))
                        origin_config_dict[hp_name] = actual_value
                        
                    elif isinstance(hp, sp.Categorical):
                        # 映射到分类选择
                        n_choices = len(hp.choices)
                        if n_choices == 1:
                            idx = 0
                        else:
                            # 将[-1,1]映射到[0, n_choices-1]
                            idx = int((normalized_value + 1) / 2 * (n_choices - 1))
                            idx = np.clip(idx, 0, n_choices - 1)
                        origin_config_dict[hp_name] = hp.choices[idx]
                        
                    else:
                        raise ValueError(f"不支持的超参数类型: {type(hp)}")
                
                result_config = Configuration(self.origin_config_space, values=origin_config_dict)
                return result_config

    def _inverse_transform_config(self, origin_config):
        """将高维原始配置转换为低维配置"""

        if self.cprs_strategy == 'ALEBO':
            # 1. 将原始配置转换为[-1,1]^D空间的数组
            if isinstance(origin_config, Configuration):
                origin_array = np.zeros(self.D)
                hp_names = self.origin_config_space.get_hyperparameter_names()
                
                for i, hp_name in enumerate(hp_names):
                    hp = self.origin_config_space.get_hyperparameter(hp_name)
                    value = origin_config[hp_name]
                    
                    if isinstance(hp, sp.Real):
                        # 映射到[-1,1]
                        normalized = 2 * (value - hp.lower) / (hp.upper - hp.lower) - 1
                        origin_array[i] = np.clip(normalized, -1.0, 1.0)
                        
                    elif isinstance(hp, sp.Integer):
                        # 映射到[-1,1]
                        normalized = 2 * (value - hp.lower) / (hp.upper - hp.lower) - 1
                        origin_array[i] = np.clip(normalized, -1.0, 1.0)
                        
                    elif isinstance(hp, sp.Categorical):
                        # 分类变量映射
                        idx = hp.choices.index(value)
                        if len(hp.choices) == 1:
                            normalized = 0.0
                        else:
                            normalized = 2 * idx / (len(hp.choices) - 1) - 1
                        origin_array[i] = normalized
                        
                    else:
                        raise ValueError(f"不支持的超参数类型: {type(hp)}")
            else:
                origin_array = np.array(origin_config)
                # 确保在[-1,1]^D范围内
                origin_array = np.clip(origin_array, -1.0, 1.0)
            
            
            # 2. 投影到低维空间
            config_array = self._alebo_project_to_low_dim(origin_array)
            # 确保config_array是一维数组
            if config_array.ndim > 1:
                config_array = config_array.flatten()

            
            # 3. 验证往返投影误差
            origin_reconstructed = self._alebo_project_to_high_dim(config_array)
            if origin_reconstructed.ndim > 1:
                origin_reconstructed = origin_reconstructed.flatten()
            roundtrip_error = np.linalg.norm(origin_array - origin_reconstructed)
            if roundtrip_error > self.alebo_projection_tolerance:
                print(f"[ALEBO] 警告：往返投影误差 = {roundtrip_error:.2e}")
            
            # 4. 转换为低维配置空间格式
            config_dict = {}
            name_len = len(str(self.cprs_k))
            for i in range(self.cprs_k):
                i_str = str(i)
                name = 'x' + '0' * (name_len - len(i_str)) + i_str
                config_dict[name] = float(config_array[i])
            
            result_config = Configuration(self.config_space, values=config_dict)
            return result_config
        
        # 如果不是ALEBO策略，返回None或其他默认值
        return None


        # =================== BAxUS核心方法 ===================
    
    def _baxus_create_projection(self):
        """创建投影矩阵"""
        np.random.seed(self.rng.randint(0, 2**31))
        self.baxus_projector = GaussianRandomProjection(
            n_components=self.target_dim, 
            random_state=self.rng.randint(0, 2**31)
        )
        dummy_data = np.eye(self.input_dim)
        if self.input_dim < self.target_dim:
            dummy_data = np.vstack([dummy_data, np.zeros((self.target_dim - self.input_dim, self.input_dim))])
        self.baxus_projector.fit(dummy_data)
        print(f"[BAxUS] Created projection to {self.target_dim} dimensions.")

    def failtol(self):
        """动态计算失败容忍度"""
        ft_max = max(4.0, self.target_dim)
        if self.target_dim == self.input_dim:
            return ft_max
            
        # 简化版本的动态计算
        evaluation_budget = 100  # 可以根据实际情况调整
        evaluation_budget = evaluation_budget - self._budget_lost_in_previous_trs
        
        _log_base = self.baxus_n_new_bins + 1
        n = max(1, round(math.log(self.input_dim / self._init_target_dim, _log_base)))
        
        budget = evaluation_budget * self.target_dim / (self._init_target_dim * n)
        
        gamma = 2 * math.log(self.baxus_length_min / self.baxus_length_init, 0.5)
        if gamma == 0:
            return ft_max
            
        ft = math.ceil(budget / gamma)
        failtol = max(1, min(ft, ft_max))
        
        return failtol

    def _budget_lost_in_previous_trs(self):
        """在之前的信任区域中消耗的预算"""
        return self.baxus_n_init if len(self._trust_region_restarts) == 0 else self._trust_region_restarts[-1]

    def _init_dim_in_tr(self):
        """当前信任区域开始时的维度"""
        if len(self._dim_in_iterations) == 0:
            return self._init_target_dim
        else:
            eval_when_tr_started = 0 if len(self._trust_region_restarts) == 0 else self._trust_region_restarts[-1]
            tr_adjust_iters = np.array(list(self._dim_in_iterations.keys()))
            relevant_iters = tr_adjust_iters[tr_adjust_iters >= eval_when_tr_started]
            if len(relevant_iters) > 0:
                min_iter = min(relevant_iters)
                return self._dim_in_iterations[min_iter]
            return self._init_target_dim

    def _sample_baxus(self, return_list=False):
        """BAxUS 主采样方法"""
        num_evaluated = len(self.history)
        
        # 1. 初始阶段：随机采样
        if num_evaluated < self.baxus_n_init:
            print(f"[BAxUS] Initial random sampling ({num_evaluated}/{self.baxus_n_init})")
            config = self.sample_random_configs(self.origin_config_space, 1)[0]
            return [config] if return_list else config

        # 2. 检查是否需要重启/分割维度
        if len(self._fX_tr) <= 1 or self.length < self.baxus_length_min:
            self._baxus_restart_or_split()

        # 3. 在当前信任区域内采样
        config = self._baxus_sample_in_tr()
        return [config] if return_list else config

    def _baxus_restart_or_split(self):
        """重启或分割维度的核心逻辑 """
        print(f"[BAxUS] eval {len(self.history)}: Restarting/splitting logic")
        
        # 检查是否可以进行维度分割
        can_split = self._can_split_dimensions()
        
        if can_split and self.target_dim < self.input_dim:
            # 执行维度分割
            self._execute_dimension_split()
        else:
            # 重启信任区域
            print(f"[BAxUS] Cannot split further. Restarting TR with new embedding.")
            self._restart_trust_region()

    def _can_split_dimensions(self):
        """检查是否可以分割维度"""
        # 简化版本：只要目标维度小于输入维度就可以分割
        return self.target_dim < self.input_dim

    def _execute_dimension_split(self):
        """执行维度分割"""
        n_evals = len(self.history)
        fbest = min([obs.objectives[0] for obs in self.history.observations 
                    if obs.objectives and len(obs.objectives) > 0])
        
        print(f"[BAxUS] eval {n_evals}: Splitting dimensions. fbest = {fbest:.4f}")
        
        # 记录分割点
        self._split_points.append(n_evals)
        self._axus_change_iterations.append(n_evals)
        
        # 增加目标维度
        old_target_dim = self.target_dim
        self.target_dim += self.baxus_n_new_bins
        self._dim_in_iterations[n_evals] = self.target_dim
        
        print(f"[BAxUS] eval {n_evals}: target dim: {old_target_dim} -> {self.target_dim}")
        
        # 重新创建投影
        self._baxus_create_projection()
        
        # 重置信任区域状态
        self.length = self.baxus_length_init
        self.failcount = 0
        self.succcount = 0
        
        # 清空TR内数据，将在下次采样时重新填充
        self._X_tr = np.empty((0, self.target_dim))
        self._fX_tr = np.empty((0, 1))

    def _restart_trust_region(self):
        """重启信任区域"""
        n_evals = len(self.history)
        
        # 记录重启点
        self._trust_region_restarts.append(n_evals)
        self._axus_change_iterations.append(n_evals)
        self._dim_in_iterations[n_evals] = self.target_dim
        
        # 重新创建投影
        self._baxus_create_projection()
        
        # 重置状态
        self.length = self.baxus_length_init
        self.failcount = 0
        self.succcount = 0
        
        # 清空TR数据
        self._X_tr = np.empty((0, self.target_dim))
        self._fX_tr = np.empty((0, 1))

    def _baxus_sample_in_tr(self):
        """在信任区域内采样"""
        # 更新TR内的数据
        self._update_tr_data()
        
        if len(self._X_tr) < 3:
            # TR内数据太少，随机采样
            print("[BAxUS] Too few points in TR, random sampling within TR")
            return self._sample_random_in_tr()
        
        # 在TR内优化采集函数 (简化版本)
        print(f"[BAxUS] Sampling in TR with {len(self._X_tr)} points, length={self.length:.4f}")
        return self._optimize_in_tr()

    def _update_tr_data(self):
        """更新信任区域内的数据"""
        self._X_tr = np.empty((0, self.target_dim))
        self._fX_tr = np.empty((0, 1))
        
        # 如果没有历史数据，直接返回
        if len(self.history) == 0:
            return
            
        # 获取当前最佳点作为TR中心
        best_config = self.history.get_incumbents()[0].config
        tr_center = self._transform_to_low_dim(best_config)
        
        # 收集TR内的所有点
        for i in range(len(self.history)):
            config = self.history.configurations[i]
            observation = self.history.observations[i]
            
            if not observation.objectives or len(observation.objectives) == 0:
                continue
            
            obj_value = observation.objectives[0]
            if obj_value is None or obj_value >= np.inf:
                continue
            
            x_low = self._transform_to_low_dim(config)
            
            # 检查是否在TR内 (使用 L∞ 范数)
            if np.all(np.abs(x_low - tr_center) <= self.length / 2):
                self._X_tr = np.vstack([self._X_tr, x_low])
                self._fX_tr = np.vstack([self._fX_tr, [[obj_value]]])

    def _sample_random_in_tr(self):
        """在TR内随机采样"""
        if len(self.history) == 0:
            return self.sample_random_configs(self.origin_config_space, 1)[0]
            
        # 获取TR中心
        best_config = self.history.get_incumbents()[0].config
        tr_center = self._transform_to_low_dim(best_config)
        
        # 在TR内随机采样
        low_dim_point = []
        for i in range(self.target_dim):
            low = max(-1, tr_center[i] - self.length / 2)
            high = min(1, tr_center[i] + self.length / 2)
            low_dim_point.append(self.rng.uniform(low, high))
        
        return self._transform_to_high_dim(np.array(low_dim_point))

    def _optimize_in_tr(self):
        """在TR内优化采集函数 """
        # 简化实现：随机采样多个候选点，选择最好的
        n_candidates = 500
        best_candidate = None
        best_score = float('inf')
        
        # 获取TR中心
        best_config = self.history.get_incumbents()[0].config
        tr_center = self._transform_to_low_dim(best_config)
        
        for _ in range(n_candidates):
            candidate = []
            for i in range(self.target_dim):
                low = max(-1, tr_center[i] - self.length / 2)
                high = min(1, tr_center[i] + self.length / 2)
                candidate.append(self.rng.uniform(low, high))
            
            # 简单评分 (可以替换为真正的采集函数)
            score = self.rng.random()
            if score < best_score:
                best_score = score
                best_candidate = candidate
        
        return self._transform_to_high_dim(np.array(best_candidate))

    def _transform_to_low_dim(self, config):
        """将高维配置转换为低维向量"""
        high_dim_vector = np.array([config[hp.name] for hp in self.origin_config_space.get_hyperparameters()])
        # 归一化到 [-1, 1]
        normalized_vector = []
        for i, hp in enumerate(self.origin_config_space.get_hyperparameters()):
            val = (high_dim_vector[i] - hp.lower) / (hp.upper - hp.lower)  # [0, 1]
            val = 2 * val - 1  # [-1, 1]
            normalized_vector.append(val)
        
        normalized_vector = np.array(normalized_vector)
        low_dim_vector = self.baxus_projector.transform(normalized_vector.reshape(1, -1))[0]
        return np.clip(low_dim_vector, -1, 1)

    def _transform_to_high_dim(self, low_dim_vector):
        """将低维向量转换为高维配置"""
        try:
            pseudo_inverse = np.linalg.pinv(self.baxus_projector.components_)
            high_dim_vector = low_dim_vector @ pseudo_inverse.T
        except:
            high_dim_vector = self.rng.uniform(-1, 1, self.input_dim)
        
        # 转换回原始范围
        new_values = {}
        for i, hp in enumerate(self.origin_config_space.get_hyperparameters()):
            if i < len(high_dim_vector):
                val = (high_dim_vector[i] + 1) / 2  # [-1, 1] -> [0, 1]
                val = val * (hp.upper - hp.lower) + hp.lower
                val = max(hp.lower, min(hp.upper, val))
            else:
                val = self.rng.uniform(hp.lower, hp.upper)
            new_values[hp.name] = val
            
        return Configuration(self.origin_config_space, values=new_values)

    def _adjust_length(self, fX_next):
        """调整信任区域长度 (参考文件逻辑)"""
        if len(self._fX_tr) == 0:
            return
            
        prev_min = np.min(self._fX_tr)
        improvement = prev_min - fX_next
        success_threshold = 0.0  # 可以调整
        
        if improvement > success_threshold:
            print(f"[BAxUS] Success! succcount: {self.succcount} -> {self.succcount + 1}")
            self.succcount += 1
            self.failcount = 0
        else:
            print(f"[BAxUS] Failure. failcount: {self.failcount} -> {self.failcount + 1}")
            self.succcount = 0
            self.failcount += 1
            
        # 调整TR长度
        if self.succcount == self.baxus_succtol:
            old_length = self.length
            self.length = min(2.0 * self.length, self.baxus_length_max)
            self.succcount = 0
            print(f"[BAxUS] TR expanded: {old_length:.4f} -> {self.length:.4f}")
            
        elif self.failcount == self.failtol:
            old_length = self.length
            self.length /= 2.0
            self.failcount = 0
            print(f"[BAxUS] TR shrunk: {old_length:.4f} -> {self.length:.4f}")

    def _sample_grvs(self):
        
        X = self.history.get_config_array()
        Y = self.history.get_objectives()

        if self.surrogate_type == 'gpf':
            self.surrogate = build_my_surrogate(func_str=self.surrogate_type, config_space=self.config_space,
                                                rng=self.rng,
                                                transfer_learning_history=self.source_hpo_data,
                                                extra_dim=self.extra_dim, norm_y=self.norm_y)
            logger.info("Successfully rebuild the surrogate model GP!")
            
        self.surrogate.train(X, Y)

        incumbent_value = self.history.get_incumbent_value()
        
        incumbent_config = self.history.get_incumbent_configs()[0]
        ori_param_names = [param.name for param in self.origin_config_space.get_hyperparameters()]
        self.acq_func.update(model=self.surrogate,
                            eta=incumbent_value,
                            num_data=len(self.history))
        
        tuned_indices = []
        
        for i in range(self.cprs_k):
            candidate_configs = []
            corre_index = []
            
            incumbent_dict = {}
            for name in ori_param_names:
                incumbent_dict[name] = incumbent_config[name]
            
            for j in range(len(self.origin_config_space)):
                if j in tuned_indices:
                    continue
                candidate_values = self.rng.uniform(-1, 1, 20)
                for k in range(20):
                    conf_dict = copy.copy(incumbent_dict)
                    conf_dict[ori_param_names[j]] = candidate_values[k]
                    candidate_configs.append(Configuration(configuration_space=self.origin_config_space, values=conf_dict))
                    corre_index.append(j)
            
            acqs = self.acq_func(candidate_configs)
            
            idx = np.argmax(acqs)
            tuned_indices.append(corre_index[idx])
            incumbent_config = candidate_configs[idx]
        
        logger.info(f"Use algorithm [{self.cprs_strategy}] to reset search space, only tune indices: {tuned_indices}")
        
        return incumbent_config

    # =================== Add-GP-UCB核心方法 ===================
    
    def _create_addgp_decomposition(self):
        """创建加性GP的维度分解"""
        total_dims = len(self.origin_config_space)
        
        # 计算需要的组数
        self.addgp_num_groups = math.ceil(total_dims / self.addgp_d_max)
        
        # 创建分解
        self.addgp_decomposition = []
        for i in range(self.addgp_num_groups):
            start_idx = i * self.addgp_d_max
            end_idx = min((i + 1) * self.addgp_d_max, total_dims)
            group_indices = list(range(start_idx, end_idx))
            self.addgp_decomposition.append(group_indices)
        
        logger.info(f"[Add-GP-UCB] Created decomposition: {self.addgp_decomposition}")

    def _addgp_extract_group_data(self, X, group_indices):
        """从完整数据中提取特定组的数据"""
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return X[:, group_indices]

    def _addgp_train_individual_gps(self, X, Y):
        """训练每个子组的GP模型"""
        self.addgp_gp_models = []
        
        # 计算共同均值函数
        common_mean = np.mean(Y) if len(Y) > 0 else 0.0
        
        for i, group_indices in enumerate(self.addgp_decomposition):
            logger.debug(f"[Add-GP-UCB] Training GP {i+1}/{self.addgp_num_groups} for dimensions {group_indices}")
            
            # 提取该组的输入数据
            X_group = self._addgp_extract_group_data(X, group_indices)
            
            # 创建核函数
            kernel = ConstantKernel(self.addgp_sigma_prs**2) * RBF(
                length_scale=self.addgp_sigma_sms, 
                length_scale_bounds=(1e-5, 1e5)
            )
            
            # 创建GP模型
            gp = GaussianProcessRegressor(
                kernel=kernel,
                alpha=self.addgp_common_noise**2,
                normalize_y=False,
                random_state=self.rng.randint(0, 2**31)
            )
            
            try:
                # 训练GP (目标值减去共同均值)
                Y_centered = Y.flatten() - common_mean
                gp.fit(X_group, Y_centered)
                self.addgp_gp_models.append(gp)
                logger.debug(f"[Add-GP-UCB] Successfully trained GP {i+1}")
            except Exception as e:
                logger.warning(f"[Add-GP-UCB] Failed to train GP {i+1}: {e}")
                # 创建一个简单的默认GP
                default_gp = GaussianProcessRegressor(
                    kernel=ConstantKernel(1.0) * RBF(1.0),
                    alpha=0.1,
                    random_state=self.rng.randint(0, 2**31)
                )
                default_gp.fit(X_group, Y_centered)
                self.addgp_gp_models.append(default_gp)
        
        # 存储共同均值
        self.addgp_common_mean = common_mean

    def _addgp_predict(self, X_test):
        """使用加性GP进行预测"""
        if X_test.ndim == 1:
            X_test = X_test.reshape(1, -1)
        
        n_test = X_test.shape[0]
        combined_mu = np.zeros(n_test)
        combined_sigma = np.zeros(n_test)
        
        # 添加共同均值
        combined_mu += self.addgp_common_mean
        
        # 预测每个子GP并求和
        for i, (gp, group_indices) in enumerate(zip(self.addgp_gp_models, self.addgp_decomposition)):
            X_group = self._addgp_extract_group_data(X_test, group_indices)
            
            try:
                mu_group, sigma_group = gp.predict(X_group, return_std=True)
                combined_mu += mu_group
                combined_sigma += sigma_group**2  # 方差相加
            except Exception as e:
                logger.warning(f"[Add-GP-UCB] Prediction failed for GP {i+1}: {e}")
                # 使用默认值
                combined_mu += 0.0
                combined_sigma += 0.1**2
        
        # 标准差是方差的平方根
        combined_sigma = np.sqrt(combined_sigma)
        
        return combined_mu, combined_sigma

    def _addgp_compute_ucb(self, X_test, num_evaluated):
        """计算Add-GP-UCB采集函数值"""
        mu, sigma = self._addgp_predict(X_test)
        
        # 计算UCB的beta参数 (基于Srinivas et al. ICML 2010)
        t = num_evaluated + 1
        D = len(self.origin_config_space)
        beta_t = self.addgp_beta_coeff * D * np.log(2 * t) / 5
        
        # UCB值
        ucb_values = mu + np.sqrt(beta_t) * sigma
        
        return ucb_values, mu, sigma

    def _sample_addgp_ucb(self, return_list=False):
        """Add-GP-UCB采样方法"""
        num_evaluated = len(self.history)
        
        # 获取历史数据
        X = self.history.get_config_array()
        Y = self.history.get_objectives()
        
        if len(X) == 0 or len(Y) == 0:
            # 没有历史数据，随机采样
            config = self.sample_random_configs(self.origin_config_space, 1)[0]
            return [config] if return_list else config
        
        # 训练加性GP模型
        logger.debug("[Add-GP-UCB] Training additive GP models...")
        self._addgp_train_individual_gps(X, Y)
        
        # 优化UCB采集函数
        logger.debug("[Add-GP-UCB] Optimizing UCB acquisition function...")
        best_config = self._addgp_optimize_ucb(num_evaluated)
        
        if return_list:
            return [best_config]
        else:
            return best_config

    def _addgp_optimize_ucb(self, num_evaluated, num_candidates=2000):
        """优化UCB采集函数找到最佳候选点"""
        # 生成候选点
        candidate_configs = self.sample_random_configs(
            self.origin_config_space, 
            num_candidates,
            excluded_configs=self.history.configurations
        )
        
        if len(candidate_configs) == 0:
            # 如果没有可用候选点，返回随机配置
            return self.sample_random_configs(self.origin_config_space, 1)[0]
        
        # 转换为数组
        X_candidates = convert_configurations_to_array(candidate_configs)
        
        # 计算UCB值
        ucb_values, mu_values, sigma_values = self._addgp_compute_ucb(X_candidates, num_evaluated)
        
        # 选择UCB值最大的点
        best_idx = np.argmax(ucb_values)
        best_config = candidate_configs[best_idx]
        
        logger.debug(f"[Add-GP-UCB] Best UCB={ucb_values[best_idx]:.4f}, "
                    f"mu={mu_values[best_idx]:.4f}, sigma={sigma_values[best_idx]:.4f}")
        
        return best_config

    def _addgp_local_search(self, start_config, num_evaluated, max_iter=50):
        """在UCB采集函数上进行局部搜索优化"""
        current_config = start_config
        current_array = convert_configurations_to_array([current_config])[0]
        current_ucb, _, _ = self._addgp_compute_ucb(current_array.reshape(1, -1), num_evaluated)
        current_ucb = current_ucb[0]
        
        for iteration in range(max_iter):
            improved = False
            
            # 为每个维度尝试小的扰动
            for dim_idx in range(len(current_array)):
                hp = self.origin_config_space.get_hyperparameter_by_idx(dim_idx)
                
                # 计算扰动范围
                if isinstance(hp, sp.Real):
                    perturbation = (hp.upper - hp.lower) * 0.1
                    for direction in [-1, 1]:
                        new_array = current_array.copy()
                        new_value = current_array[dim_idx] + direction * perturbation
                        new_value = max(hp.lower, min(hp.upper, new_value))
                        new_array[dim_idx] = new_value
                        
                        # 创建新配置
                        try:
                            new_values = {}
                            for i, param in enumerate(self.origin_config_space.get_hyperparameters()):
                                new_values[param.name] = new_array[i]
                            new_config = Configuration(self.origin_config_space, values=new_values)
                            
                            # 检查是否已经评估过
                            if new_config in self.history.configurations:
                                continue
                                
                            # 计算UCB值
                            new_ucb, _, _ = self._addgp_compute_ucb(new_array.reshape(1, -1), num_evaluated)
                            new_ucb = new_ucb[0]
                            
                            # 如果更好，更新当前点
                            if new_ucb > current_ucb:
                                current_config = new_config
                                current_array = new_array
                                current_ucb = new_ucb
                                improved = True
                                break
                        except Exception as e:
                            logger.debug(f"[Add-GP-UCB] Local search failed for dim {dim_idx}: {e}")
                            continue
                
                if improved:
                    break
            
            # 如果没有改进，停止搜索
            if not improved:
                break
        
        logger.debug(f"[Add-GP-UCB] Local search completed after {iteration+1} iterations, "
                    f"final UCB={current_ucb:.4f}")
        
        return current_config
    

      # =================== PP-GP-UCB核心方法 (基于projection-Add-GP.pdf) ===================
    
    def _learn_projection_matrices_ppgp(self, X, y):
        """
        学习组投影矩阵W_list = {W^(1), ..., W^(M)}
        基于Gilboa et al. (2013)的贪心策略，逐轮优化投影矩阵
        """
        if len(X) < 3:
            logger.warning("[PP-GP-UCB] Not enough data for projection learning")
            return False
            
        try:
            from scipy.optimize import minimize
            
            N, D = X.shape
            
            # 标准化数据
            self.ppgp_X_mean = np.mean(X, axis=0)
            self.ppgp_X_std = np.std(X, axis=0) + 1e-8
            X_normalized = (X - self.ppgp_X_mean) / self.ppgp_X_std
            
            self.ppgp_W_list = []
            self.ppgp_gps = []
            residuals = y.copy()  # 残差初始化（用于贪心迭代）
            
            logger.debug(f"[PP-GP-UCB] Learning {self.ppgp_M} projection matrices...")
            
            for j in range(self.ppgp_M):
                logger.debug(f"[PP-GP-UCB] Learning projection matrix {j+1}/{self.ppgp_M}")
                
                # 1. 初始化当前投影矩阵W^(j)（D×d维）
                W_j = self._init_group_projection(X_normalized, residuals)
                
                if W_j is None:
                    logger.warning(f"[PP-GP-UCB] Failed to initialize projection matrix {j+1}")
                    break
                
                # 2. 优化W^(j)和对应GP的核超参数（最大化边际似然）
                W_opt, gp = self._optimize_group_projection(X_normalized, residuals, W_j)
                
                if W_opt is None or gp is None:
                    logger.warning(f"[PP-GP-UCB] Failed to optimize projection matrix {j+1}")
                    break
                
                # 3. 投影数据到d维子空间
                Z_j = X_normalized @ W_opt  # (N, d)
                
                # 4. 训练当前子空间的GP模型
                try:
                    gp.fit(Z_j, residuals)
                    predictions = gp.predict(Z_j)
                    
                    # 5. 更新残差（减去当前GP的预测值）
                    residuals = residuals - predictions.flatten()
                    
                    # 6. 保存投影矩阵和GP模型
                    self.ppgp_W_list.append(W_opt)
                    self.ppgp_gps.append(gp)
                    
                    logger.debug(f"[PP-GP-UCB] Successfully learned projection matrix {j+1}")
                    
                except Exception as e:
                    logger.warning(f"[PP-GP-UCB] Failed to train GP for projection {j+1}: {e}")
                    break
            
            logger.debug(f"[PP-GP-UCB] Learned {len(self.ppgp_W_list)} projection matrices")
            return len(self.ppgp_W_list) > 0
            
        except Exception as e:
            logger.error(f"[PP-GP-UCB] Failed to learn projection matrices: {e}")
            return False

    def _init_group_projection(self, X, residuals):
        """初始化组投影矩阵W^(j)（D×d维）：基于残差的线性回归+随机化"""
        try:
            D = X.shape[1]
            
            # 基于线性回归初始化
            if len(residuals) >= D and np.var(residuals) > 1e-8:
                try:
                    # 使用线性回归的主要方向作为初始化
                    from sklearn.linear_model import Ridge
                    ridge = Ridge(alpha=0.1)
                    
                    # 为了生成d维投影，我们创建d个轻微扰动的目标
                    W = np.zeros((D, self.ppgp_d))
                    
                    for i in range(self.ppgp_d):
                        # 添加小的随机噪声创建不同的目标
                        noise = self.rng.normal(0, 0.1 * np.std(residuals), len(residuals))
                        target = residuals + noise
                        
                        ridge.fit(X, target)
                        w = ridge.coef_
                        
                        # 归一化并正交化
                        w = w / (np.linalg.norm(w) + 1e-8)
                        
                        # Gram-Schmidt正交化
                        for k in range(i):
                            w = w - np.dot(w, W[:, k]) * W[:, k]
                        w = w / (np.linalg.norm(w) + 1e-8)
                        
                        W[:, i] = w
                        
                except Exception:
                    # 降级为随机初始化
                    W = self.rng.randn(D, self.ppgp_d)
            else:
                # 随机初始化
                W = self.rng.randn(D, self.ppgp_d)
            
            # 列归一化和正交化
            for i in range(self.ppgp_d):
                # 正交化
                for j in range(i):
                    W[:, i] = W[:, i] - np.dot(W[:, i], W[:, j]) * W[:, j]
                # 归一化
                W[:, i] = W[:, i] / (np.linalg.norm(W[:, i]) + 1e-8)
            
            return W
            
        except Exception as e:
            logger.debug(f"[PP-GP-UCB] Failed to initialize group projection: {e}")
            return None

    def _optimize_group_projection(self, X, y, W_init):
        """
        优化组投影矩阵W^(j)和核超参数
        目标：最大化GP的边际似然（低维子空间上的GP）
        """
        try:
            from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
            
            def neg_marginal_likelihood(params):
                """负边际似然（需最小化）"""
                try:
                    # 解析参数：前D*d个为W^(j)的元素，剩余为核超参数
                    D, d = W_init.shape
                    W_flat = params[:D * d]
                    W = W_flat.reshape(D, d)
                    
                    # 正交化投影矩阵
                    W, _ = np.linalg.qr(W)
                    W = W[:, :d]  # 确保维度正确
                    
                    # 核超参数
                    length_scale = np.exp(params[D * d])  # 标量长度尺度
                    noise = np.exp(params[D * d + 1])     # 噪声方差
                    
                    # 投影到d维子空间
                    Z = X @ W  # (N, d)
                    
                    # 定义核函数
                    kernel = ConstantKernel(1.0) * Matern(
                        length_scale=length_scale, 
                        nu=1.5
                    ) + WhiteKernel(noise_level=noise)
                    
                    # 计算负边际似然
                    gp = GaussianProcessRegressor(
                        kernel=kernel, 
                        alpha=0,
                        random_state=self.rng.randint(0, 2**31)
                    )
                    gp.fit(Z, y)
                    return -gp.log_marginal_likelihood(gp.kernel_.theta)
                    
                except Exception:
                    return 1e6  # 拟合失败时返回大的惩罚值
            
            # 初始参数：W的扁平化 + 核超参数（长度尺度+噪声）
            init_params = np.concatenate([
                W_init.flatten(),
                [np.log(1.0)],      # 长度尺度初始值
                [np.log(0.01)]      # 噪声初始值
            ])
            
            # 优化（L-BFGS-B）
            result = minimize(
                neg_marginal_likelihood, 
                init_params, 
                method='L-BFGS-B',
                options={'maxiter': 100}
            )
            
            if not result.success:
                logger.debug("[PP-GP-UCB] Optimization failed, using initial values")
                W_opt = W_init.copy()
                length_scale = 1.0
                noise = 0.01
            else:
                # 解析优化结果
                D, d = W_init.shape
                W_opt = result.x[:D * d].reshape(D, d)
                
                # 正交化优化后的投影矩阵
                W_opt, _ = np.linalg.qr(W_opt)
                W_opt = W_opt[:, :d]
                
                length_scale = np.exp(result.x[D * d])
                noise = np.exp(result.x[D * d + 1])
            
            # 构建优化后的GP模型
            optimized_kernel = ConstantKernel(1.0) * Matern(
                length_scale=length_scale, 
                nu=1.5
            ) + WhiteKernel(noise_level=noise)
            
            optimized_gp = GaussianProcessRegressor(
                kernel=optimized_kernel, 
                alpha=0,
                random_state=self.rng.randint(0, 2**31)
            )
            
            return W_opt, optimized_gp
            
        except Exception as e:
            logger.debug(f"[PP-GP-UCB] Failed to optimize group projection: {e}")
            return None, None

    def _select_next_point_ppgp(self):
        """
        在低维子空间用Add-GP-UCB选点并反投影到高维
        每个子空间独立优化UCB，组合后反投影
        """
        try:
            if len(self.ppgp_W_list) == 0 or len(self.ppgp_gps) == 0:
                return None
            
            # 获取历史数据
            X = self.history.get_config_array()
            if len(X) == 0:
                return None
                
            X_normalized = (X - self.ppgp_X_mean) / self.ppgp_X_std
            
            # 生成低维候选点（每个子空间独立采样）
            z_candidates_list = []
            
            for j in range(len(self.ppgp_W_list)):
                W_j = self.ppgp_W_list[j]
                gp_j = self.ppgp_gps[j]
                
                # 子空间j的投影数据范围
                Z_j = X_normalized @ W_j
                z_min, z_max = Z_j.min(axis=0), Z_j.max(axis=0)
                
                # 扩展搜索范围
                z_range = z_max - z_min
                z_min = z_min - 0.2 * z_range
                z_max = z_max + 0.2 * z_range
                
                # 在子空间j中生成候选点
                n_candidates = 200
                z_candidates = self.rng.uniform(
                    z_min, z_max, size=(n_candidates, self.ppgp_d)
                )
                
                # 计算UCB = 均值 + beta*标准差
                mu, sigma = gp_j.predict(z_candidates, return_std=True)
                
                # 计算beta参数
                t = len(self.history) + 1
                beta = 2.0 * self.ppgp_d * np.log(2 * t)  # 基于理论保证
                
                ucb = mu + np.sqrt(beta) * sigma
                
                # 选择子空间j的最优候选点
                best_idx = np.argmax(ucb)
                z_best_j = z_candidates[best_idx]
                z_candidates_list.append(z_best_j)
                
                logger.debug(f"[PP-GP-UCB] Subspace {j+1}: best UCB={ucb[best_idx]:.4f}")
            
            # 反投影到高维空间（用每个投影矩阵的伪逆组合）
            x_candidates = []
            for j in range(len(self.ppgp_W_list)):
                W_j = self.ppgp_W_list[j]
                z_best_j = z_candidates_list[j]
                
                # 伪逆映射：x = W_j
                try:
                    W_pinv = np.linalg.pinv(W_j)
                    x_j = W_pinv @ z_best_j
                    x_candidates.append(x_j)
                except Exception as e:
                    logger.debug(f"[PP-GP-UCB] Failed to compute pseudo-inverse for matrix {j}: {e}")
                    continue
            
            if len(x_candidates) == 0:
                return None
            
            # 取平均作为最终高维候选点（简化处理）
            x_next_normalized = np.mean(x_candidates, axis=0)
            
            # 反标准化
            x_next = x_next_normalized * self.ppgp_X_std + self.ppgp_X_mean
            
            # 转换为配置
            config_dict = {}
            for i, hp in enumerate(self.origin_config_space.get_hyperparameters()):
                if i < len(x_next):
                    value = float(x_next[i])
                    # 确保在超参数边界内
                    if hasattr(hp, 'lower') and hasattr(hp, 'upper'):
                        value = max(hp.lower, min(hp.upper, value))
                    config_dict[hp.name] = value
                else:
                    # 如果维度不够，使用随机值
                    if hasattr(hp, 'lower') and hasattr(hp, 'upper'):
                        config_dict[hp.name] = self.rng.uniform(hp.lower, hp.upper)
                    else:
                        config_dict[hp.name] = hp.default_value
            
            return Configuration(self.origin_config_space, values=config_dict)
            
        except Exception as e:
            logger.error(f"[PP-GP-UCB] Failed to select next point: {e}")
            return None

    def _sample_ppgp_ucb(self, return_list=False):
        """PP-GP-UCB主采样方法"""
        num_evaluated = len(self.history)
        
        # 获取历史数据
        X = self.history.get_config_array()
        Y = self.history.get_objectives()
        
        # 初始阶段：随机采样
        if len(X) < self.ppgp_num_init:
            config = self.sample_random_configs(self.origin_config_space, 1)[0]
            logger.debug(f"[PP-GP-UCB] Initial random sampling ({len(X)}/{self.ppgp_num_init})")
            return [config] if return_list else config
        
        # 学习/更新投影矩阵（每N_cyc次迭代）
        if (len(self.ppgp_W_list) == 0 or 
            num_evaluated % self.ppgp_N_cyc == 0):
            logger.debug("[PP-GP-UCB] Learning projection matrices using PP-GP...")
            success = self._learn_projection_matrices_ppgp(X, Y)
            if not success:
                # 学习失败，降级为随机采样
                config = self.sample_random_configs(self.origin_config_space, 1)[0]
                logger.warning("[PP-GP-UCB] Projection learning failed, using random sampling")
                return [config] if return_list else config
        
        # 在低维空间中优化并反投影
        logger.debug("[PP-GP-UCB] Optimizing in projected subspaces...")
        best_config = self._select_next_point_ppgp()
        
        if best_config is None:
            # 选点失败，降级为随机采样
            logger.warning("[PP-GP-UCB] Point selection failed, using random sampling")
            best_config = self.sample_random_configs(self.origin_config_space, 1)[0]
        
        # 检查是否已经评估过
        if best_config in self.history.configurations:
            logger.debug("[PP-GP-UCB] Generated duplicate config, using random sampling")
            best_config = self.sample_random_configs(
                self.origin_config_space, 1, 
                excluded_configs=self.history.configurations
            )[0]
        
        logger.debug("[PP-GP-UCB] Generated candidate point")
        
        if return_list:
            return [best_config]
        else:
            return best_config


    # =================== RPP-GP-UCB核心方法 ===================
    
    def _sample_rpp_gp_ucb(self, return_list=False):
        """RPP-GP-UCB采样方法"""
        num_evaluated = len(self.history)
        
        # 获取历史数据
        X = self.history.get_config_array()
        Y = self.history.get_objectives()
        
        if len(X) == 0 or len(Y) == 0:
            # 没有历史数据，随机采样
            config = self.sample_random_configs(self.origin_config_space, 1)[0]
            return [config] if return_list else config
        
        # 每N_cyc次迭代更新带约束的投影矩阵
        if num_evaluated % self.rpp_N_cyc == 1 or len(self.rpp_W_hat_list) == 0:
            logger.debug(f"[RPP-GP-UCB] Learning restricted projection matrices at iteration {num_evaluated}")
            self._rpp_learn_restricted_projections(X, Y)
        
        # 在外盒域中优化UCB采集函数
        logger.debug("[RPP-GP-UCB] Optimizing UCB in outer bounding box...")
        best_config = self._rpp_optimize_ucb_in_outer_box(X, num_evaluated)
        
        if return_list:
            return [best_config]
        else:
            return best_config
    
    def _rpp_compute_volume_ratio(self, W_j, X):
        """计算外盒域体积与有效域体积的比值（δ-ratio）"""
        if X.shape[0] < 2:
            return 1.0  # 数据太少时返回默认值
            
        # 投影数据
        Z_proj = X @ W_j  # (N, d)
        
        # 计算外盒域体积
        z_min, z_max = Z_proj.min(axis=0), Z_proj.max(axis=0)
        outer_volume = np.prod(z_max - z_min + 1e-8)
        
        # 有效域体积近似（使用凸包体积）
        try:
            from scipy.spatial import ConvexHull
            if Z_proj.shape[0] > Z_proj.shape[1]:  # 确保点数大于维度数
                hull = ConvexHull(Z_proj)
                effective_volume = hull.volume
            else:
                effective_volume = outer_volume
        except:
            effective_volume = outer_volume
        
        # 体积比
        ratio = outer_volume / (effective_volume + 1e-12)
        return ratio
    
    def _rpp_learn_restricted_projections(self, X, Y):
        """学习带δ-ratio约束的投影矩阵（与PP-GP-UCB使用相同的贪心学习策略）"""
        if len(X) < 3:
            logger.warning("[RPP-GP-UCB] Not enough data for projection learning")
            return False
        
        try:
            from scipy.optimize import minimize
            
            N, D = X.shape
            
            # 标准化数据（与PP-GP-UCB一致）
            self.rpp_X_mean = np.mean(X, axis=0)
            self.rpp_X_std = np.std(X, axis=0) + 1e-8
            X_normalized = (X - self.rpp_X_mean) / self.rpp_X_std
            
            self.rpp_W_hat_list = []
            self.rpp_gp_models = []
            self.rpp_common_mean = np.mean(Y)
            residuals = Y.flatten() - self.rpp_common_mean  # 使用残差（与PP-GP-UCB一致）
            
            logger.debug(f"[RPP-GP-UCB] Learning {self.rpp_M} projection matrices with delta-ratio constraint...")
            
            for j in range(self.rpp_M):
                logger.debug(f"[RPP-GP-UCB] Learning projection matrix {j+1}/{self.rpp_M}")
                
                # 1. 初始化投影矩阵（与PP-GP-UCB相同的方法）
                W_j = self._rpp_init_group_projection(X_normalized, residuals)
                
                if W_j is None:
                    logger.warning(f"[RPP-GP-UCB] Failed to initialize projection matrix {j+1}")
                    break
                
                # 2. 优化投影矩阵和核超参数（增加δ-ratio约束）
                W_opt, gp = self._rpp_optimize_group_projection_with_constraint(
                    X_normalized, residuals, W_j
                )
                
                if W_opt is None or gp is None:
                    logger.warning(f"[RPP-GP-UCB] Failed to optimize projection matrix {j+1}")
                    break
                
                # 3. 投影数据到d维子空间
                Z_j = X_normalized @ W_opt  # (N, d)
                
                # 4. 训练当前子空间的GP模型
                try:
                    gp.fit(Z_j, residuals)
                    predictions = gp.predict(Z_j)
                    
                    # 5. 更新残差（与PP-GP-UCB一致的贪心策略）
                    residuals = residuals - predictions.flatten()
                    
                    # 6. 保存投影矩阵和GP模型
                    self.rpp_W_hat_list.append(W_opt)
                    self.rpp_gp_models.append(gp)
                    
                    logger.debug(f"[RPP-GP-UCB] Successfully learned projection matrix {j+1}")
                    
                except Exception as e:
                    logger.warning(f"[RPP-GP-UCB] Failed to train GP for projection {j+1}: {e}")
                    break
            
            # 保存当前投影矩阵用于下一轮warm start
            self.rpp_prev_W = self.rpp_W_hat_list.copy()
            logger.debug(f"[RPP-GP-UCB] Learned {len(self.rpp_W_hat_list)} projection matrices")
            return len(self.rpp_W_hat_list) > 0
            
        except Exception as e:
            logger.error(f"[RPP-GP-UCB] Failed to learn projection matrices: {e}")
            return False

    def _rpp_init_group_projection(self, X, residuals):
        """初始化组投影矩阵（与PP-GP-UCB完全相同）"""
        return self._init_group_projection(X, residuals)  # 直接复用PP-GP-UCB的方法

    
    def _rpp_optimize_group_projection_with_constraint(self, X, y, W_init):
        """
        优化组投影矩阵W^(j)和核超参数（在PP-GP-UCB基础上增加δ-ratio约束）
        """
        try:
            from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
            
            def neg_marginal_likelihood_with_constraint(params):
                """负边际似然 + δ-ratio约束惩罚"""
                try:
                    # 解析参数
                    D, d = W_init.shape
                    W_flat = params[:D * d]
                    W = W_flat.reshape(D, d)
                    
                    # 正交化投影矩阵
                    W, _ = np.linalg.qr(W)
                    W = W[:, :d]
                    
                    # 计算δ-ratio约束违反度（RPP-GP-UCB的核心改进）
                    ratio = self._rpp_compute_volume_ratio(W, X)
                    constraint_violation = max(0, ratio - (1 + self.rpp_delta)) * 1e3
                    
                    # 核超参数
                    length_scale = np.exp(params[D * d])
                    noise = np.exp(params[D * d + 1])
                    
                    # 投影到子空间并计算边际似然
                    Z = X @ W
                    kernel = ConstantKernel(1.0) * Matern(
                        length_scale=length_scale, nu=1.5
                    ) + WhiteKernel(noise_level=noise)
                    
                    gp = GaussianProcessRegressor(
                        kernel=kernel, alpha=0,
                        random_state=self.rng.randint(0, 2**31)
                    )
                    gp.fit(Z, y)
                    nll = -gp.log_marginal_likelihood(gp.kernel_.theta)
                    
                    return nll + constraint_violation  # 边际似然 + 约束惩罚
                    
                except Exception:
                    return 1e6
                
            # 与PP-GP-UCB相同的优化策略
            init_params = np.concatenate([
                W_init.flatten(),
                [np.log(1.0)],      # 长度尺度
                [np.log(0.01)]      # 噪声
            ])
        
            result = minimize(
            neg_marginal_likelihood_with_constraint, 
            init_params, 
            method='L-BFGS-B',
            options={'maxiter': 100}
            )

            if not result.success:
                logger.debug("[RPP-GP-UCB] Optimization failed, using initial values")
                W_opt = W_init.copy()
                length_scale = 1.0
                noise = 0.01
            else:
                D, d = W_init.shape
                W_opt = result.x[:D * d].reshape(D, d)
                W_opt, _ = np.linalg.qr(W_opt)
                W_opt = W_opt[:, :d]
                length_scale = np.exp(result.x[D * d])
                noise = np.exp(result.x[D * d + 1])

             # 验证约束
            final_ratio = self._rpp_compute_volume_ratio(W_opt, X)
            if final_ratio > (1 + self.rpp_delta):
                logger.warning(f"[RPP-GP-UCB] Volume ratio {final_ratio:.2f} exceeds constraint")
            
            # 构建GP模型
            optimized_kernel = ConstantKernel(1.0) * Matern(
            length_scale=length_scale, nu=1.5
            ) + WhiteKernel(noise_level=noise)
        
            optimized_gp = GaussianProcessRegressor(
                kernel=optimized_kernel, alpha=0,
                random_state=self.rng.randint(0, 2**31)
            )
            
            return W_opt, optimized_gp
            
        except Exception as e:
            logger.debug(f"[RPP-GP-UCB] Failed to optimize with constraint: {e}")
            return None, None

    def _rpp_optimize_ucb_in_outer_box(self, X, num_evaluated):
        """在外盒域中优化UCB采集函数"""
        if len(self.rpp_W_hat_list) == 0 or len(self.rpp_gp_models) == 0:
            return self.sample_random_configs(self.origin_config_space, 1)[0]
        
        z_candidates_list = []
        ucb_scores = []
        
        # 为每个子空间在其外盒域中选择最优点
        for j in range(len(self.rpp_W_hat_list)):
            W_j = self.rpp_W_hat_list[j]
            gp_j = self.rpp_gp_models[j]
            
            # 计算外盒域outer(Ŵ)
            Z_j = X @ W_j
            z_min, z_max = Z_j.min(axis=0), Z_j.max(axis=0)
            
            # 在外盒域内生成候选点
            n_candidates = 200
            z_candidates = self.rng.uniform(
                z_min, z_max, 
                size=(n_candidates, self.rpp_d_max)
            )
            
            # 计算UCB值
            try:
                mu, sigma = gp_j.predict(z_candidates, return_std=True)
                # UCB参数
                t = num_evaluated + 1
                beta_t = self.rpp_beta_coeff * self.rpp_d_max * np.log(2 * t) / 5
                ucb = mu + np.sqrt(beta_t) * sigma
                
                # 选择当前子空间的最优候选点
                best_idx = np.argmax(ucb)
                z_best_j = z_candidates[best_idx]
                z_candidates_list.append(z_best_j)
                ucb_scores.append(ucb[best_idx])
            except Exception as e:
                logger.warning(f"[RPP-GP-UCB] UCB computation failed for subspace {j}: {e}")
                # 使用随机点作为后备
                z_candidates_list.append(self.rng.uniform(z_min, z_max, size=self.rpp_d_max))
                ucb_scores.append(0.0)
        
        # 反投影到高维：使用加权平均（权重为UCB分数）
        if len(z_candidates_list) == 0:
            return self.sample_random_configs(self.origin_config_space, 1)[0]
        
        # 归一化UCB分数作为权重
        ucb_scores = np.array(ucb_scores)
        if np.sum(ucb_scores) > 0:
            weights = ucb_scores / np.sum(ucb_scores)
        else:
            weights = np.ones(len(ucb_scores)) / len(ucb_scores)
        
        x_candidates = []
        for j in range(len(z_candidates_list)):
            W_j = self.rpp_W_hat_list[j]
            z_best_j = z_candidates_list[j]
            
            # 伪逆反投影
            try:
                W_pinv = np.linalg.pinv(W_j)
                x_j = W_pinv @ z_best_j
                x_candidates.append(x_j)
            except:
                # 后备：使用最小二乘解
                x_j = np.linalg.lstsq(W_j, z_best_j, rcond=None)[0]
                x_candidates.append(x_j)
        
        # 加权平均反投影结果
        x_next = np.average(x_candidates, axis=0, weights=weights)
        
        # 转换为配置对象
        try:
            # 裁剪到参数边界
            config_values = {}
            for i, param in enumerate(self.origin_config_space.get_hyperparameters()):
                if i < len(x_next):
                    if hasattr(param, 'lower') and hasattr(param, 'upper'):
                        value = np.clip(x_next[i], param.lower, param.upper)
                    else:
                        value = x_next[i]
                    config_values[param.name] = value
                else:
                    config_values[param.name] = param.default_value
            
            from ConfigSpace import Configuration
            config = Configuration(self.origin_config_space, values=config_values)
            
            # 检查是否已经评估过
            if config in self.history.configurations:
                # 如果重复，返回随机配置
                config = self.sample_random_configs(self.origin_config_space, 1)[0]
            
            return config
            
        except Exception as e:
            logger.warning(f"[RPP-GP-UCB] Configuration creation failed: {e}")
            return self.sample_random_configs(self.origin_config_space, 1)[0]

