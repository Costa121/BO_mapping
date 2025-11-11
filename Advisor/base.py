from ConfigSpace import ConfigurationSpace, Configuration
import numpy as np
import json as js
import copy
import pickle as pkl
from ConfigSpace.read_and_write.json import write
from .utils import map_source_hpo_data, build_observation
from openbox import logger
from openbox.utils.history import History
from ConfigSpace import ConfigurationSpace


MAXINT = 2 ** 31 - 1


class BaseAdvisor:
    def __init__(self, config_space: ConfigurationSpace, source_hpo_data=None,
                 task_id='test', ep_args=None, ep_strategy='none', context_flag=False,
                 ws_strategy='none', ws_args=None, tl_args=None, cprs_k=35,
                 cprs_strategy='none', space_history = None, meta_feature=None,
                 seed=42, rng=None, rand_prob=0.15, rand_mode='ran', **kwargs):

        self._logger_kwargs = kwargs.get('_logger_kwargs', None)

        # self.his_X, self.his_y = space_history if space_history else [None, None]
        self.cprs_k = cprs_k

        # 随机种子
        self.seed = seed
        self.rand_prob = rand_prob
        self.rand_mode = rand_mode
        if rng is None:
            rng = np.random.RandomState(self.seed)
        self.rng = rng
        
        self.origin_config_space = config_space
        self.config_space = config_space
        self.fixed_indices = None
        self.config_space_seed = self.rng.randint(MAXINT)

        # 历史数据
        self.context_flag = context_flag
        self.last_context = meta_feature['ini_context']
        meta_feature['cprs_k'] = cprs_k
        meta_feature['seed'] = seed
        meta_feature['rand_prob'] = rand_prob
        meta_feature['rand_mode'] = rand_mode
        meta_feature['ori_space'] = js.loads(write(config_space))
        self.history = History(task_id=task_id, config_space=self.config_space, meta_info=meta_feature)
        self.origin_history = History(task_id=task_id, config_space=self.origin_config_space, meta_info=meta_feature)

        self.task_id = task_id
        self.ws_strategy = ws_strategy
        self.ws_args = ws_args
        self.tl_args = tl_args
        self.ep_args = ep_args
        self.ep_strategy = ep_strategy
        self.cprs_strategy = cprs_strategy
        self.meta_feature = meta_feature

        self.source_hpo_data_sims = None
        self.source_hpo_data = self.filter_source_hpo_data(source_hpo_data)


    def filter_source_hpo_data(self, source_hpo_data):

        if source_hpo_data is None or len(source_hpo_data) == 0:
            logger.info("No source hpo data provided.")
            return list()

        map_strategy = self.ws_strategy
        new_source_hpo_data = list()
        if map_strategy in ['none', 'best_all']:
            sims = [[i, 0.5] for i in range(len(source_hpo_data))]
        else:
            sims = map_source_hpo_data(map_strategy=map_strategy,
                                    target_his=self.history, source_hpo_data=source_hpo_data, **self.ws_args)
        tl_topk = len(source_hpo_data)
        if self.tl_args['topk'] > 0:
            tl_topk = self.tl_args['topk']

        self.source_hpo_data_sims = []  # 重新排序
        for i in range(tl_topk):
            new_source_hpo_data.append(source_hpo_data[sims[i][0]])
            self.source_hpo_data_sims.append([i, sims[i][1]])
            logger.info("The %d-th similar task(%s): %s" % (sims[i][0], source_hpo_data[sims[i][0]].task_id, sims[i][1]))

        logger.info("Successfully filter source hpo data to %d tasks." % len(new_source_hpo_data))

        return new_source_hpo_data

    @property
    def contexts(self):
        contexts = self.history.observations[0]['last_context']
        for idx in range(1, len(self.history)):
            contexts = np.vstack([contexts, self.history.observations[idx]['last_context']])
        return contexts


    def warm_start(self):

        raise NotImplementedError

    def sample(self):
        raise NotImplementedError

    @staticmethod
    def sample_random_configs(config_space, num_configs=1, excluded_configs=None):
        """
        Sample a batch of random configurations.

        Parameters
        ----------
        config_space: ConfigurationSpace
            Configuration space object.
        num_configs: int
            Number of configurations to sample.
        excluded_configs: optional, List[Configuration] or Set[Configuration]
            A list of excluded configurations.

        Returns
        -------
        configs: List[Configuration]
            A list of sampled configurations.
        """
        if excluded_configs is None:
            excluded_configs = set()

        configs = list()
        sample_cnt = 0
        max_sample_cnt = 1000
        while len(configs) < num_configs:
            config = config_space.sample_configuration()
            sample_cnt += 1
            if config not in configs and config not in excluded_configs:
                config.origin = "Random Sample!"
                configs.append(config)
                sample_cnt = 0
                continue
            if sample_cnt >= max_sample_cnt:
                logger.warning('Cannot sample non duplicate configuration after %d iterations.' % max_sample_cnt)
                config.origin = "Random Sample(duplicate)!"
                configs.append(config)
                sample_cnt = 0
        return configs

    def update(self, config, results):
        obs = build_observation(config, results, last_context=self.last_context)
        self.history.update_observation(obs)
        self.last_context = results['result']['context']

