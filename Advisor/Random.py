from .base import BaseAdvisor

from ConfigSpace import ConfigurationSpace


class Random(BaseAdvisor):
    def __init__(self, config_space: ConfigurationSpace, task_id='test',
                 meta_feature=None,
                 seed=42, rng=None):
        super().__init__(config_space, task_id=task_id,
                         meta_feature=meta_feature,
                         seed=seed, rng=rng)
        
        # 初始化ini_configs属性
        self.ini_configs = list()

    def update(self, config, results):
        super(Random, self).update(config, results)

    def warm_start(self):

        raise NotImplementedError

    """
    采样(使用ini_configs进行热启动和普通采样)
    以及安全约束 (40轮后阈值为 0.85 * incumbent_value)
    """

    def sample(self, return_list=False):
        if len(self.ini_configs) > 0:
            config = self.ini_configs[-1]
            self.ini_configs.pop()
        else:
            config = self.sample_random_configs(self.config_space, 1,
                                                excluded_configs=self.history.configurations)[0]
        if return_list:
            return [config]
        else:
            return config
