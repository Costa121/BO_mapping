from .base import BaseOptimizer
from math import log, ceil
from openbox import logger
import numpy as np
from scipy.stats import kendalltau
from build_utils import run_obj_func
from openbox.utils.config_space.util import convert_configurations_to_array


class BOHB(BaseOptimizer):
    def __init__(self, config_space, eval_func, iter_num=200, per_run_time_limit=None,
                 source_hpo_data=None, ep_args=None, ep_strategy='none',
                 method_id='smbo', task_id='test', target='redis', task_str='run',
                 space_history = None, cprs_strategy='none',
                 ws_strategy='none', ws_args=None, tl_strategy='none', tl_args=None,
                 backup_flag=False, save_dir='./results', meta_feature=None,
                 R=9, eta=3, seed=42, rand_prob=0.15, rand_mode='ran',
                 fgf_flag=False, revive_flag=False, flex_flag=False, k_thres=0.5, config_modifier = None):

        super().__init__(config_space=config_space, eval_func=eval_func, iter_num=iter_num,
                         per_run_time_limit=per_run_time_limit, source_hpo_data=source_hpo_data,
                         method_id=method_id, task_id=task_id, target=target, task_str=task_str,
                         ws_strategy=ws_strategy, ws_args=ws_args, tl_strategy=tl_strategy, tl_args=tl_args,
                         ep_args=ep_args, ep_strategy=ep_strategy,
                         cprs_strategy=cprs_strategy, space_history=space_history,
                         backup_flag=backup_flag, save_dir=save_dir, meta_feature=meta_feature,
                         seed=seed, rand_prob=rand_prob, rand_mode=rand_mode, config_modifier = config_modifier)

        self.R = R
        self.eta = eta
        self.rand_prob = rand_prob

        self.logeta = lambda x: log(x) / log(self.eta)
        self.s_max = int(self.logeta(self.R))
        self.B = (self.s_max + 1) * self.R
        self.s_values = list(reversed(range(self.s_max + 1)))
        self.inner_iter_id = 0

        self.full_eval_configs = list()
        self.full_eval_perfs = list()

        self.advisor.history.meta_info['R_eta'] = "%d_%d" % (self.R, self.eta)
        self._initialize_params(fgf_flag, revive_flag, flex_flag, k_thres)
        

    def _initialize_params(self, fgf_flag, revive_flag, flex_flag, k_thres):
        self.fgf_flag = fgf_flag
        if not fgf_flag:
            self.fidelity_levels = np.logspace(0, self.s_max, self.s_max + 1, base=self.eta)
        else:
            self.fidelity_levels = [1] + list(range(self.eta, self.R + self.eta, self.eta))
        self.fidelity_levels = [int(x) for x in self.fidelity_levels]
        logger.info("Fidelity levels of %s: %s" % (self.method_id, self.fidelity_levels))

        # Setting for GloSH
        # λ shall increase as fidelity level grows.
        # with R = 27, for r0 = 1, ..., r3 = 27, s_max = 3, we set λ0 = 1/3, λ1 = 1/2, λ2 = 1.
        # with R = 81, for r0 = 1, ..., r4 = 81, s_max = 4, we set λ0 = 1/4, λ1 = 1/3, λ2 = 1/2, λ3 = 1.
        self.revive_flag = revive_flag
        if revive_flag:
            self.revive_weights = dict()
            for i in range(self.s_max):
                self.revive_weights[int(self.eta ** i)] = 1 / (self.s_max - i)
    
        self.flex_flag = flex_flag
        self.flexible_n = list()
        self.flexible_r = list()
        for s in self.s_values:  # Same as bohb at the beginning
            n = int(ceil(self.B / self.R / (s + 1) * self.eta ** s)) 
            r = int(self.R * self.eta ** (-s))
            self.flexible_n.append(n)
            self.flexible_r.append(r)
        if flex_flag:
            self.flex_collected = False
            self.flex_collected_perfs = list()
            self.k_thres = k_thres  # Threshold for rank correlation
        
        logger.info("Run %d brackets with number of configurations %s and resources %s." 
                    % (len(self.flexible_n), self.flexible_n, self.flexible_r))


    def _gen_candidates(self, num_configs):
        if 'BOHB' in self.method_id:
            cans = self.advisor.sample(return_list=True)

            candidates = list()
            idx_acq = 0
            for _id in range(num_configs):
                if self.rng.random() < self.rand_prob or _id >= len(cans):
                    i = 0
                    config = self.advisor.sample_random_configs(
                        self.advisor.config_space, 1, excluded_configs=self.advisor.history.configurations)[0]
                    while config in candidates:
                        config = self.advisor.sample_random_configs(
                            self.advisor.config_space, 1, excluded_configs=self.advisor.history.configurations)[0]
                        i += 1
                        if i > 1000:
                            logger.warning('Cannot sample a new random configuration after 1000 iters.')
                            break
                    config.origin = 'BOHB Random Sample'
                else:
                    config = cans[idx_acq]
                    config.origin = 'BOHB ' + config.origin
                    idx_acq += 1
                candidates.append(config)
            return candidates
        elif 'MFSE' in self.method_id or 'FlexHB' in self.method_id:
            return self.advisor.samples(batch_size=num_configs)
        else:
            raise ValueError('Invalid method_id: %s!' % self.method_id)

    def _update_obs(self, config, results, resource_ratio, special_list=None):
        if 'BOHB' in self.method_id:
            if resource_ratio == 1:
                self.advisor.update(config=config, results=results)
        elif 'MFSE' in self.method_id or 'FlexHB' in self.method_id:
            if special_list is None:
                self.advisor.update(config=config, results=results, resource_ratio=resource_ratio)
            else:
                pass
        else:
            raise ValueError('Invalid method_id: %s!' % self.method_id)
        
    def _run_flexband(self):
        n_prime = self.flexible_n[:]
        r_prime = self.flexible_r[:]
        logger.info(str(self.flex_collected_perfs))

        for j in range(len(self.flexible_n) - 1, 0, -1):    # j => [ s_max, 1 ]
            i = j - 1   # i => [ s_max - 1, 0 ]
            
            x1 = self.flex_collected_perfs[j]
            x2 = self.flex_collected_perfs[i]
            if len(x1) < 2 or len(x2) < 2:
                logger.warn("Skipping kendall's tau computation due to insufficient data: x1=%s, x2=%s" % (x1, x2))
                continue
            tau, _ = kendalltau(x1, x2)
            if tau is not None and not np.isnan(tau):  # 避免 tau 为 nan
                logger.info("Get value of tau: %f" % (tau))
                if tau > self.k_thres:
                    n_prime[j] = self.flexible_n[i]
                    r_prime[j] = self.flexible_r[i]
            else:
                logger.warn("Tau computation resulted in NaN, skipping update")

        self.flexible_n = n_prime
        self.flexible_r = r_prime
        logger.info("Update initial number of configurations %s and resource %s of each brackets."
                    % (self.flexible_n, self.flexible_r))
    
    
    def _handle_intermediate_fidelities(self, candidates, n_resource, visit_fidelities):
        # Handle intermediate fidelities if the fgf_flag is set
        if self.fgf_flag:
            intermediate_fidelities = [
                i for i in self.fidelity_levels if i < n_resource and not visit_fidelities[i]
            ]
            logger.info("Add intermediate fidelities: %s for fidelity-%d." % (intermediate_fidelities, n_resource))
            for inter_fide in intermediate_fidelities:
                visit_fidelities[inter_fide] = True 
                inter_res_ratio = float(inter_fide / self.R)
                _ = self._evaluate_configurations(candidates, inter_res_ratio)
        return


    def _evaluate_configurations(self, candidates, resource_ratio, update=True):
        # Evaluate the configurations and update observations
        performances = []
        for config in candidates:
            obj_args, obj_kwargs = (config,), dict(resource_ratio=resource_ratio)
            results = run_obj_func(self.eval_func, obj_args, obj_kwargs, self.timeout)
            if update:
                self._update_obs(config, results, resource_ratio=resource_ratio)
            performances.append(results['result']['objective'])
        return performances
                

    def _iterate(self, s, n, r, skip_last=0):
        # Set initial number of configurations
        # n = int(ceil(self.B / self.R / (s + 1) * self.eta ** s))  # (smax+1) / (s+1) * 3^s
        # initial number of iterations per config
        # r = int(self.R * self.eta ** (-s))                        # 81 * 3^(-s)

        # Choose a batch of configurations in different mechanisms.
        candidates = list(set(self._gen_candidates(n)))

        iter_full_eval_configs, iter_full_eval_perfs = [], []
        
        visit_fidelities = {i: log(i, self.eta).is_integer() for i in self.fidelity_levels}

        # Run each bracket.
        for i in range((s + 1) - int(skip_last)):  # changed from s + 1
            # Run each of the n configs for <iterations>
            # and keep best (n_configs / eta) configurations
            n_configs = n * self.eta ** (-i)
            n_resource = r * self.eta ** i

            resource_ratio = float(n_resource / self.R)

            # Handling revived candidates if revive_flag is set
            revived_candidates, early_stop_configs = [], []
            idx = self.advisor.get_resource_index(resource_ratio)
            if self.revive_flag and idx >= 0 and int(n_resource) != self.R:
                early_stop_configs = self.advisor.early_stop_histories[idx][:]
                self.advisor.early_stop_histories[idx].clear()
                revived_candidates.extend(early_stop_configs)
                logger.info(f"Index of resource {resource_ratio:.5f} is {idx}. Revived {len(revived_candidates)} configs.")
            
            # Evaluate intermediate fidelities
            self._handle_intermediate_fidelities(candidates, n_resource, visit_fidelities)

            # Evaluate configurations
            revived_candidates.extend(candidates)
            perfs = self._evaluate_configurations(revived_candidates, resource_ratio)

            logger.info('iter: %d, resource: %d, val_loss: %s' % (i, n_resource, str(perfs)))
            
            # Select a number of best configurations for the next loop.
            # Filter out early stops, if any.
            indices = np.argsort(perfs)
            revived_candidates = [revived_candidates[i] for i in indices]
            perfs = [perfs[i] for i in indices]

            candidates_keep, perfs_keep = [], []
            revive_prob = self.revive_weights.get(int(n_resource), 0)
            for config, perf in zip(revived_candidates, perfs):
                if config in early_stop_configs and self.rng.random() < revive_prob:
                    logger.warn("Revive an early stop configuration in fidelity-%d, probability-%f" 
                                % (n_resource, self.revive_weights.get(int(n_resource), 0)))
                    candidates_keep.append(config)
                    perfs_keep.append(perf)
                elif config in candidates:
                    candidates_keep.append(config)
                    perfs_keep.append(perf)
                if len(candidates_keep) == n_configs:
                    break

            if len(candidates_keep) >= self.eta:
                reduced_num = int(n_configs / self.eta)
                candidates, perfs = candidates_keep[:reduced_num], perfs_keep[:reduced_num]
            else:
                candidates, perfs = [candidates_keep[0]], [perfs_keep[0]]
            throw_configs = [config for config in revived_candidates if config not in candidates]
            
            if int(n_resource) != self.R:
                idx = self.advisor.get_resource_index(resource_ratio)
                self.advisor.early_stop_histories[idx].extend(throw_configs)
            
                logger.info("Get [%d] C_keep and [%d] C_throw, [%d] remain to be validate" 
                            % (len(candidates_keep), len(throw_configs), len(candidates)))
                logger.info("Update [%d] early stop configurations of resource [%d]." 
                        % (len(self.advisor.early_stop_histories[idx]), n_resource))
            else:
                iter_full_eval_configs.extend(candidates)
                iter_full_eval_perfs.extend(perfs)

        return iter_full_eval_configs, iter_full_eval_perfs


    def run_one_iter(self):

        self.iter_id += 1
        num_config_evaluated = len(self.advisor.history)
        if num_config_evaluated < self.advisor.init_num:
            candidates = self.advisor.sample(return_list=True)
            assert len(candidates) == 1
            config = candidates[0]
            obj_args, obj_kwargs = (config,), dict(resource_ratio=1)
            results = run_obj_func(self.eval_func, obj_args, obj_kwargs, self.timeout)
            self._update_obs(config, results, resource_ratio=1)
            iter_full_eval_configs = candidates
            iter_full_eval_perfs = [results['result']['objective']]
        elif self.flex_flag and not len(self.flex_collected_perfs):
            test_configs = self.advisor.sample_random_configs(self.advisor.config_space, self.R,
                                                              excluded_configs=self.advisor.history.configurations)
            for r in self.flexible_r:
                perfs = self._evaluate_configurations(test_configs, float(r / self.R), update=False)
                self.flex_collected_perfs.append(perfs)
            self._run_flexband()
            return
        else:
            iter_full_eval_configs, iter_full_eval_perfs = self._iterate(
                self.s_values[self.inner_iter_id],
                self.flexible_n[self.inner_iter_id],
                self.flexible_r[self.inner_iter_id]
            )
            self.inner_iter_id = (self.inner_iter_id + 1) % (self.s_max + 1)

        logger.info(
            "[{}] iter ------------------------------------------------{:5d}".format(self.task_str, self.iter_id))
        for idx, config in enumerate(iter_full_eval_configs):
            if config.origin:
                logger.warn("[{}] !!!!!!!!!! {} !!!!!!!!!!".format(self.task_str, config.origin))

            logger.info('[{}] Config: '.format(self.task_str) + str(config.get_dictionary()))
            logger.info('[{}] Obj: {}'.format(self.task_str, iter_full_eval_perfs[idx]))

        logger.info('[{}] best obj: {}'.format(self.task_str, self.advisor.history.get_incumbent_value()))

        logger.info(
            "[{}] ===================================================================================================================================================".format(
                self.task_str))

        self.save_info(interval=self.s_max)






