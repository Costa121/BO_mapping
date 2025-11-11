from .base import BaseOptimizer
from Advisor.tas_bo import TASBOAdvisor
from openbox import logger  # 添加这行导入


class TASBOOptimizer(BaseOptimizer):
    def __init__(self, config_space, eval_func, iter_num=200, per_run_time_limit=None,
                 source_hpo_data=None, method_id='TAS-BO', task_id='test', target='test',
                 task_str='run', meta_feature=None, ws_strategy='none', ws_args=None,
                 tl_strategy='none', tl_args=None, backup_flag=False, save_dir='./results',
                 local_num_factor=2.0, seed=42, rand_prob=0.15, **kwargs):
        
        super().__init__(
            config_space=config_space, eval_func=eval_func, iter_num=iter_num,
            per_run_time_limit=per_run_time_limit, source_hpo_data=source_hpo_data,
            method_id=method_id, task_id=task_id, target=target, task_str=task_str,
            ws_strategy=ws_strategy, ws_args=ws_args, tl_strategy=tl_strategy,
            tl_args=tl_args, backup_flag=backup_flag, save_dir=save_dir,
            seed=seed, **kwargs
        )
        
        # 初始化TAS-BO advisor
        self.advisor = TASBOAdvisor(
            config_space=config_space,
            source_hpo_data=source_hpo_data,
            task_id=self.task_id,
            meta_feature=meta_feature,
            ws_strategy=ws_strategy,
            ws_args=ws_args,
            tl_args=tl_args,
            local_num_factor=local_num_factor,
            seed=seed,
            rng=self.rng,
            rand_prob=rand_prob,
            **kwargs
        )
        
        self.timeout = per_run_time_limit
        
    def run_one_iter(self):
        """运行一次迭代"""
        self.iter_id += 1
        
        # 获取下一个配置
        config = self.advisor.sample()
        
        # 评估配置
        obj_args, obj_kwargs = (config,), dict()
        from build_utils import run_obj_func
        results = run_obj_func(self.eval_func, obj_args, obj_kwargs, self.timeout)
        
        # 更新观察历史
        self.advisor.update(config, results)
        
        # 保存信息 - 确保每次迭代都保存
        self.save_info(interval=1)  # 每次都保存
        
        # 日志输出
        current_best = self.advisor.history.get_incumbent_value()
        logger.info(f"[{self.task_str}] Iteration {self.iter_id}: "
                   f"current best = {current_best:.6f}")
        
        # 显示保存路径
        if hasattr(self, 'result_path'):
            logger.info(f"[{self.task_str}] Results saving to: {self.result_path}")
        
        return config, results