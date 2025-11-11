from .base import BaseAdvisor
from .utils import build_my_surrogate, build_my_acq_func
from .acq_optimizer.context_local_random import InterleavedLocalAndRandomSearch

import numpy as np
import copy
from ConfigSpace import Configuration, ConfigurationSpace
from openbox import logger
from openbox import space as sp
from openbox.utils.config_space.util import convert_configurations_to_array
from openbox.utils.history import Observation, History
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler


class TASBOAdvisor(BaseAdvisor):
    def __init__(self, config_space: ConfigurationSpace, source_hpo_data=None,
                 surrogate_type='prf', acq_type='ei', task_id='test', meta_feature=None,
                 ws_strategy='none', ws_args={'init_num': 5}, tl_args={'topk': 5},
                 local_num_factor=2.0, seed=42, rng=None, rand_prob=0.15, **kwargs):
        """
        TAS-BO实现
        Args:
            local_num_factor: 局部邻近点数量因子，默认为2倍维度
        """
        # 确保正确传递 meta_feature 参数
        super().__init__(config_space, task_id=task_id, meta_feature=meta_feature, 
                        source_hpo_data=source_hpo_data, 
                        ws_strategy=ws_strategy, ws_args=ws_args, tl_args=tl_args,
                        seed=seed, rng=rng, rand_prob=rand_prob, **kwargs)
        
        # 确保 ini_configs 属性存在
        if not hasattr(self, 'ini_configs'):
            self.ini_configs = []
        
        self.surrogate_type = surrogate_type
        self.acq_type = acq_type
        self.ws_strategy = ws_strategy
        self.ws_args = ws_args
        self.tl_args = tl_args
        self.source_hpo_data = source_hpo_data
        self.meta_feature = meta_feature
        
        # 添加初始化数量属性
        self.init_num = ws_args.get('init_num', 5)
        
        # TAS-BO特有参数
        self.local_num_factor = local_num_factor
        self.local_num = int(local_num_factor * len(config_space))
        
        # 随机数生成器
        if rng is None:
            self.rng = np.random.RandomState(seed)
        else:
            self.rng = rng
        self.rand_prob = rand_prob
        
        # 初始化代理模型
        self.global_surrogate = None
        self.local_surrogate = None
        self.global_acq_func = None
        self.local_acq_func = None
        
        # 采集函数优化器
        self.global_acq_optimizer = None
        self.local_acq_optimizer = None
        
        # 数据标准化器
        self.scaler = StandardScaler()
        
        # 初始化模型
        self._initialize_models()
        
        logger.info(f"[TAS-BO] Initialized with local_num={self.local_num}, "
                   f"dimensions={len(config_space)}")
    
    def _initialize_models(self):
        """初始化全局和局部代理模型"""
        # 全局代理模型
        self.global_surrogate = build_my_surrogate(
            func_str=self.surrogate_type, 
            config_space=self.config_space, 
            rng=self.rng,
            transfer_learning_history=self.source_hpo_data
        )
        
        # 局部代理模型
        self.local_surrogate = build_my_surrogate(
            func_str=self.surrogate_type,
            config_space=self.config_space,
            rng=self.rng
        )
        
        # 全局采集函数
        self.global_acq_func = build_my_acq_func(
            func_str=self.acq_type,
            model=self.global_surrogate
        )
        
        # 局部采集函数
        self.local_acq_func = build_my_acq_func(
            func_str=self.acq_type,
            model=self.local_surrogate
        )
        
        # 采集函数优化器
        self.global_acq_optimizer = InterleavedLocalAndRandomSearch(
            acquisition_function=self.global_acq_func,
            config_space=self.config_space,
            rng=self.rng,
            rand_prob=self.rand_prob
        )
        
        self.local_acq_optimizer = InterleavedLocalAndRandomSearch(
            acquisition_function=self.local_acq_func,
            config_space=self.config_space,
            rng=self.rng,
            rand_prob=self.rand_prob
        )
    
    def sample_random_configs(self, config_space, num_configs, excluded_configs=None):
        """随机采样配置"""
        if excluded_configs is None:
            excluded_configs = []
        
        configs = []
        max_attempts = num_configs * 100  # 防止无限循环
        attempts = 0
        
        while len(configs) < num_configs and attempts < max_attempts:
            config = config_space.sample_configuration()
            if config not in excluded_configs:
                configs.append(config)
            attempts += 1
        
        # 如果仍然不够，使用默认配置填充
        while len(configs) < num_configs:
            configs.append(config_space.get_default_configuration())
        
        return configs

    def sample(self, return_list=False):
        """TAS-BO采样方法"""
        num_config_evaluated = len(self.history)
        
        # 初始化阶段：随机采样或热启动
        if num_config_evaluated < self.init_num:
            if len(self.ini_configs) > 0:
                config = self.ini_configs.pop()
            else:
                config = self.sample_random_configs(
                    self.config_space, 1, 
                    excluded_configs=self.history.configurations
                )[0]
                
            if return_list:
                return [config]
            else:
                return config
        
        # TAS-BO主要采样逻辑
        try:
            # 步骤1：训练全局高斯过程模型
            global_best_config = self._train_global_model()
            
            # 步骤2：在全局最优点附近训练局部高斯过程模型
            local_best_config = self._train_local_model(global_best_config)
            
            # 步骤3：返回局部优化结果
            if return_list:
                return [local_best_config]
            else:
                return local_best_config
                
        except Exception as e:
            logger.warning(f"[TAS-BO] Sampling failed: {e}, using random configuration")
            config = self.sample_random_configs(
                self.config_space, 1,
                excluded_configs=self.history.configurations
            )[0]
            
            if return_list:
                return [config]
            else:
                return config
    
    def _train_global_model(self):
        """训练全局高斯过程模型并找到全局最优点"""
        # 获取历史数据
        X = self.history.get_config_array()
        Y = self.history.get_objectives()
        
        if len(X) == 0:
            return self.sample_random_configs(self.config_space, 1)[0]
        
        # 训练全局代理模型
        self.global_surrogate.train(X, Y)
        
        # 更新全局采集函数
        incumbent_value = self.history.get_incumbent_value()
        self.global_acq_func.update(
            model=self.global_surrogate,
            eta=incumbent_value,
            num_data=len(self.history)
        )
        
        # 优化全局采集函数找到最佳点
        challengers = self.global_acq_optimizer.maximize(
            observations=self.history.observations,
            num_points=1000
        )
        
        global_best_config = challengers.challengers[0]
        
        logger.debug(f"[TAS-BO] Global optimization completed, "
                    f"found best config with acquisition value")
        
        return global_best_config
    
    def _train_local_model(self, global_best_config):
        """在全局最优点附近训练局部高斯过程模型"""
        # 获取历史数据
        X = self.history.get_config_array()
        Y = self.history.get_objectives()
        
        if len(X) < self.local_num:
            logger.warning(f"[TAS-BO] Insufficient data for local model "
                          f"({len(X)} < {self.local_num}), using global result")
            return global_best_config
        
        # 找到离全局最优点最近的点
        global_best_array = convert_configurations_to_array([global_best_config])[0]
        distances = cdist([global_best_array], X, metric='euclidean')[0]
        
        # 选择最近的local_num个点
        nearest_indices = np.argsort(distances)[:self.local_num]
        local_X = X[nearest_indices]
        local_Y = Y[nearest_indices]
        
        # 创建局部配置空间（基于局部数据的边界）
        local_config_space = self._create_local_config_space(local_X)
        
        # 重新初始化局部模型和优化器
        self.local_surrogate = build_my_surrogate(
            func_str=self.surrogate_type,
            config_space=local_config_space,
            rng=self.rng
        )
        
        self.local_acq_func = build_my_acq_func(
            func_str=self.acq_type,
            model=self.local_surrogate
        )
        
        self.local_acq_optimizer = InterleavedLocalAndRandomSearch(
            acquisition_function=self.local_acq_func,
            config_space=local_config_space,
            rng=self.rng,
            rand_prob=self.rand_prob
        )
        
        # 训练局部代理模型
        self.local_surrogate.train(local_X, local_Y)
        
        # 更新局部采集函数
        incumbent_value = np.min(local_Y)
        self.local_acq_func.update(
            model=self.local_surrogate,
            eta=incumbent_value,
            num_data=len(local_Y)
        )
        
        # 创建局部观察历史
        local_observations = []
        for i, idx in enumerate(nearest_indices):
            original_obs = self.history.observations[idx]
            # 将配置转换到局部空间
            local_config = self._convert_to_local_config(
                self.history.configurations[idx], local_config_space
            )
            
            local_obs = Observation(
                config=local_config,
                objectives=original_obs.objectives,
                constraints=getattr(original_obs, 'constraints', None),
                trial_state=original_obs.trial_state,
                elapsed_time=original_obs.elapsed_time
            )
            local_observations.append(local_obs)
        
        # 在局部空间中优化采集函数
        challengers = self.local_acq_optimizer.maximize(
            observations=local_observations,
            num_points=500
        )
        
        local_best_config = challengers.challengers[0]
        
        # 将局部配置转换回原始空间
        final_config = self._convert_to_original_config(
            local_best_config, local_config_space
        )
        
        logger.debug(f"[TAS-BO] Local optimization completed in region "
                    f"with {len(local_X)} points")
        
        return final_config
    
    def _create_local_config_space(self, local_X):
        """基于局部数据创建局部配置空间"""
        local_config_space = ConfigurationSpace()
        
        # 计算每个维度的边界
        mins = np.min(local_X, axis=0)
        maxs = np.max(local_X, axis=0)
        
        # 为每个超参数创建局部版本
        for i, hp in enumerate(self.config_space.get_hyperparameters()):
            if isinstance(hp, sp.Real):
                # 确保边界在原始范围内
                local_lower = max(hp.lower, float(mins[i]))
                local_upper = min(hp.upper, float(maxs[i]))
                
                # 避免边界相等
                if local_lower >= local_upper:
                    local_lower = hp.lower
                    local_upper = hp.upper
                
                local_hp = sp.Real(
                    name=f"local_{hp.name}",
                    lower=local_lower,
                    upper=local_upper,
                    default_value=(local_lower + local_upper) / 2
                )
                
            elif isinstance(hp, sp.Integer):
                local_lower = max(hp.lower, int(np.floor(mins[i])))
                local_upper = min(hp.upper, int(np.ceil(maxs[i])))
                
                if local_lower >= local_upper:
                    local_lower = hp.lower
                    local_upper = hp.upper
                
                local_hp = sp.Integer(
                    name=f"local_{hp.name}",
                    lower=local_lower,
                    upper=local_upper,
                    default_value=(local_lower + local_upper) // 2
                )
                
            elif isinstance(hp, sp.Categorical):
                # 分类变量保持不变
                local_hp = sp.Categorical(
                    name=f"local_{hp.name}",
                    choices=hp.choices,
                    default_value=hp.default_value
                )
            else:
                # 其他类型保持不变
                local_hp = copy.deepcopy(hp)
                local_hp.name = f"local_{hp.name}"
            
            local_config_space.add_hyperparameter(local_hp)
        
        return local_config_space
    
    def _convert_to_local_config(self, original_config, local_config_space):
        """将原始配置转换为局部配置"""
        local_values = {}
        
        for hp in local_config_space.get_hyperparameters():
            original_name = hp.name.replace("local_", "")
            if original_name in original_config:
                local_values[hp.name] = original_config[original_name]
            else:
                local_values[hp.name] = hp.default_value
        
        return Configuration(local_config_space, values=local_values)
    
    def _convert_to_original_config(self, local_config, local_config_space):
        """将局部配置转换为原始配置"""
        original_values = {}
        
        for hp in self.config_space.get_hyperparameters():
            local_name = f"local_{hp.name}"
            if local_name in local_config:
                original_values[hp.name] = local_config[local_name]
            else:
                original_values[hp.name] = hp.default_value
        
        return Configuration(self.config_space, values=original_values)
    
    def warm_start(self):
        """热启动逻辑"""
        if self.ws_strategy == 'none' or self.source_hpo_data is None:
            return
        
        # 简单的热启动实现
        logger.info(f"[TAS-BO] Starting warm start with {len(self.source_hpo_data)} source tasks")
        
        # 从最佳源任务中获取初始配置
        for source_history in self.source_hpo_data[:self.tl_args.get('topk', 3)]:
            if len(source_history.observations) > 0:
                best_obs = min(source_history.observations, 
                             key=lambda x: x.objectives[0] if x.objectives else float('inf'))
                
                # 尝试将配置适配到当前空间
                try:
                    adapted_config = self._adapt_config(best_obs.config)
                    if adapted_config is not None:
                        adapted_config.origin = f"TAS-BO-WS-{source_history.task_id}"
                        self.ini_configs.append(adapted_config)
                        
                        if len(self.ini_configs) >= self.ws_args.get('init_num', 5):
                            break
                except Exception as e:
                    logger.debug(f"[TAS-BO] Failed to adapt config: {e}")
                    continue
    
    def _adapt_config(self, source_config):
        """适配源配置到目标空间"""
        try:
            adapted_values = {}
            for hp in self.config_space.get_hyperparameters():
                if hp.name in source_config:
                    adapted_values[hp.name] = source_config[hp.name]
                else:
                    adapted_values[hp.name] = hp.default_value
            
            return Configuration(self.config_space, values=adapted_values)
        except:
            return None