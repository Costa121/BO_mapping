import copy
from typing import List
from .BO import BO
from openbox import logger
from openbox.utils.history import History, Observation
from ConfigSpace import Configuration, ConfigurationSpace

from .utils import build_observation


class MFBO(BO):
    def __init__(self, config_space: ConfigurationSpace, source_hpo_data=None,
                 surrogate_type='prf', acq_type='ei', task_id='test', meta_feature=None,
                 ws_strategy='none', ws_args={'init_num': 5}, tl_args={'topk': 5},
                 cprs_strategy='none', space_history = None, ep_args=None, ep_strategy='none',
                 safe_flag=False, seed=42, rng=None, rand_prob=0.15, rand_mode='ran', **kwargs):
        super().__init__(config_space, source_hpo_data=source_hpo_data,
                         surrogate_type=surrogate_type, acq_type=acq_type, task_id=task_id,
                         ws_strategy=ws_strategy, ws_args=ws_args, tl_args=tl_args,
                         ep_args=ep_args, ep_strategy=ep_strategy, meta_feature=meta_feature,
                         cprs_strategy=cprs_strategy, space_history=space_history,
                         safe_flag=safe_flag, seed=seed, rng=rng, rand_prob=rand_prob, rand_mode=rand_mode, **kwargs)

        self.history_list: List[History] = list()  # 低精度组的 history -> List[History]
        self.early_stop_histories: List[List] = list()  # 早停被抛弃没有被复活的配置
        self.resource_identifiers = list()
        if self.source_hpo_data is not None and not surrogate_type.startswith('mfse'):
            self.history_list = self.source_hpo_data
            self.resource_identifiers = [-1] * len(self.source_hpo_data)  # 占位符


    """
    采样(使用ini_configs进行热启动和普通采样)
    以及安全约束 (40轮后阈值为 0.85 * incumbent_value)
    """
    def samples(self, batch_size):
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
            return [config]

        self.surrogate.update_mf_trials(self.history_list)
        self.surrogate.build_source_surrogates()
        candidates = super().sample(return_list=True)  # super().sample一定已经过了热启动阶段

        batch_configs_list = list()
        idx = 0
        while len(batch_configs_list) < batch_size:
            if self.rng.random() < self.rand_prob or idx >= len(candidates):
                # sample random configuration proportionally
                logger.info('Sample random config. rand_prob=%f.' % self.rand_prob)
                cur_config = self.sample_random_configs(
                    self.config_space, 1, excluded_configs=self.history.configurations + batch_configs_list)[0]
                cur_config.origin = 'MFSE Random Sample'
            else:
                cur_config = None
                while idx < len(candidates):
                    conf = candidates[idx]
                    idx += 1
                    if conf not in batch_configs_list and conf not in self.history.configurations:
                        cur_config = conf
                        break
            if cur_config is not None:
                batch_configs_list.append(cur_config)

        return batch_configs_list

    def update(self, config, results, resource_ratio=1):
        context = results['result']['context']
        obs = build_observation(config, results, last_context=self.last_context)
        resource_ratio = round(resource_ratio, 5)
        if resource_ratio != 1:
            if resource_ratio not in self.resource_identifiers:
                self.resource_identifiers.append(resource_ratio)
                history = History(task_id="res%.5f_%s" % (resource_ratio, self.task_id), num_objectives=self.history.num_objectives,
                                  num_constraints=self.history.num_constraints,
                                  config_space=self.config_space)
                self.history_list.append(history)
                self.early_stop_histories.append([])
            self.history_list[self.get_resource_index(resource_ratio)].update_observation(obs)

        else:
            self.history.update_observation(obs)
            self.last_context = context
            
        # if self.context_flag:
        #     self.acq_optimizer.update_contex(context)
        #     print("successfully update context!")


    def get_resource_index(self, resource_ratio):
        rounded_ratio = round(resource_ratio, 5)
        try:
            return self.resource_identifiers.index(rounded_ratio)
        except ValueError:
            return -1
