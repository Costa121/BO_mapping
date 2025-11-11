import numpy as np

from openbox import logger
import json
import time
import traceback
from multiprocessing import Process, Queue
import threading
from queue import Empty


SUCCESS = 0
FAILED = 1
TIMEOUT = 2
MEMOUT = 3


def build_optimizer(args, **kwargs):
    task_str = kwargs.get('task_str', 'run')
    meta_feature = kwargs.get('meta_feature', None)
    source_hpo_data = kwargs.get('source_hpo_data', None)
    source_compress_data = kwargs.get('source_compress_data', None)

    ws_args = kwargs.get('ws_args', None)
    tl_args = kwargs.get('tl_wargs', None)
    ep_args = kwargs.get('ep_args', None)

    per_run_time_limit = kwargs.get('per_run_time_limit', None)
    config_modifier = kwargs.get('config_modifier', None)

    # 确保config_space被定义
    if 'config_space' not in kwargs:
        from test_function import get_config_space  # 或者其他获取配置空间的方式
        config_space = get_config_space(args.func, args.dim)
    else:
        config_space = kwargs['config_space']

    optimizer = None
    if args.opt == 'RS':
        from Optimizer.smbo import SMBO
        optimizer = SMBO(config_space=kwargs['config_space'], eval_func=kwargs['eval_func'],
                         iter_num=args.iter_num, per_run_time_limit=per_run_time_limit, meta_feature=meta_feature,
                         method_id='RS', task_id=args.task, target=kwargs['target'], task_str=task_str, config_modifier=config_modifier)
    if args.opt in ['GP', 'GPF', 'SMAC']:
        from Optimizer.smbo import SMBO
        optimizer = SMBO(
            config_space=kwargs['config_space'], eval_func=kwargs['eval_func'],
            iter_num=args.iter_num, per_run_time_limit=per_run_time_limit, meta_feature=meta_feature,
            source_hpo_data=source_hpo_data, ep_args=ep_args, ep_strategy=args.expert,
            method_id=args.opt, task_id=args.task, target=kwargs['target'], task_str=task_str,
            cprs_strategy=args.compress, space_history=source_compress_data, cprs_k = args.cprs_k,
            ws_strategy=args.warm_start, ws_args=ws_args, tl_strategy=args.transfer, tl_args=tl_args,
            backup_flag=args.backup_flag, seed=args.seed, rand_prob=args.rand_prob, rand_mode=args.rand_mode, config_modifier=config_modifier
        )
    elif 'BOHB' in args.opt or 'MFSE' in args.opt or 'FlexHB' in args.opt:
        from Optimizer.bohb import BOHB
        R = args.R
        eta = args.eta
        _flexhb_args = kwargs.get('flexhb_args', {
            'fgf_flag': True, 'revive_flag': True, 'flex_flag': True, 'k_thres': 0.5
        })  # 默认全开
        flexhb_args = _flexhb_args if 'FlexHB' in args.opt else {}
        logger.warning("Args for FLEXHB_SMAC: %s" % (flexhb_args))
        optimizer = BOHB(
            config_space=kwargs['config_space'], eval_func=kwargs['eval_func'], meta_feature=meta_feature,
            iter_num=args.iter_num, per_run_time_limit=per_run_time_limit, source_hpo_data=source_hpo_data,
            ep_args=ep_args, ep_strategy=args.expert, cprs_strategy=args.compress, space_history=source_compress_data,
            method_id=args.opt, task_id=args.task, target=kwargs['target'], task_str=task_str,
            ws_strategy=args.warm_start, ws_args=ws_args, tl_strategy=args.transfer, tl_args=tl_args,
            backup_flag=args.backup_flag, seed=args.seed, rand_prob=args.rand_prob, rand_mode=args.rand_mode,
            R=R, eta=eta, config_modifier=config_modifier, **flexhb_args
        )

    elif args.opt == 'TPE':
        from Advisor.tpe import TPEAdvisor
        optimizer = TPEAdvisor(
            config_space=kwargs['config_space'], eval_func=kwargs['eval_func'], iter_num=args.iter_num,
            task_id=args.task,
        )
    elif args.opt == 'TAS-BO':
        from Optimizer.tas_bo_optimizer import TASBOOptimizer
        
        # 确保meta_feature有正确的结构
        if meta_feature is None:
            meta_feature = {'ini_context': None}
        elif not isinstance(meta_feature, dict):
            meta_feature = {'ini_context': None}
        elif 'ini_context' not in meta_feature:
            meta_feature['ini_context'] = None
        
        # 确保ws_args存在
        if ws_args is None:
            ws_args = {'init_num': 5}
        
        # 设置保存目录
        save_dir = './results'
        
        optimizer = TASBOOptimizer(
            config_space=config_space,
            eval_func=kwargs['eval_func'],
            iter_num=args.iter_num,
            per_run_time_limit=per_run_time_limit,
            source_hpo_data=source_hpo_data,
            method_id='TAS-BO',
            task_id=args.task,
            target=kwargs['target'],
            task_str=task_str,
            meta_feature=meta_feature,
            ws_strategy=args.warm_start,
            ws_args=ws_args,
            tl_strategy=args.transfer,
            tl_args=tl_args,
            backup_flag=args.backup_flag,
            save_dir=save_dir,  # 显式设置保存目录
            seed=args.seed,
            rand_prob=args.rand_prob,
            local_num_factor=2.0
        )
    elif args.compress == 'RPP-GP-UCB':
        # RPP-GP-UCB作为压缩策略使用
        kwargs['cprs_strategy'] = 'RPP-GP-UCB'
        kwargs['cprs_k'] = args.cprs_k
        # RPP-GP-UCB特定参数
        kwargs['rpp_d_max'] = min(3, len(config_space))
        kwargs['rpp_M'] = min(5, len(config_space) // kwargs['rpp_d_max'])
        kwargs['rpp_delta'] = 0.2
        kwargs['rpp_N_cyc'] = 10
        optimizer = build_bo_optimizer(args, **kwargs)
    logger.info("[opt: {}] [warm_start_strategy: {}] [transfer_strategy: {}] [safe_flag: {}] [backup_flag: {}] [tasks: {}] [seed: {}] [rand_prob: {}]".format(
        args.opt, args.warm_start, args.transfer, args.safe_flag, args.backup_flag, task_str, args.seed, args.rand_prob)
    )

    logger.info("warm start args: %s: %s" % (args.warm_start, json.dumps(ws_args)))


    return optimizer


def wrapper_func(obj_func, queue, obj_args, obj_kwargs):
    try:
        ret = obj_func(*obj_args, **obj_kwargs)
    except Exception:
        result = {'result': {'objective': np.Inf, 'context': None}, 'timeout': False, 'traceback': traceback.format_exc()}
    else:
        result = {'result': ret, 'timeout': False, 'traceback': None}
    queue.put(result)


def _check_result(result):
    if isinstance(result, dict) and set(result.keys()) == {'result', 'timeout', 'traceback'}:
        return result
    else:
        return {'result': {'objective': np.Inf, 'context': None}, 'timeout': True, 'traceback': None}


def run_without_time_limit(obj_func, obj_args, obj_kwargs):
    start_time = time.time()
    try:
        ret = obj_func(*obj_args, **obj_kwargs)
    except Exception:
        result = {'result': {'objective': np.Inf, 'context': None}, 'timeout': False, 'traceback': traceback.format_exc()}
    else:
        if np.isfinite(ret['objective']):
            result = {'result': ret, 'timeout': False, 'traceback': None}
        else:
            result = {'result': ret, 'timeout': True, 'traceback': None}
    result['elapsed_time'] = time.time() - start_time
    return result


def run_with_time_limit(obj_func, obj_args, obj_kwargs, timeout):
    start_time = time.time()
    queue = Queue()
    # Todo: 这里本来是多进程，但是多进程传入的参数要是可序列化的（SparkExecutor不满足），所以暂时改成多线程
    p = Process(target=wrapper_func, args=(obj_func, queue, obj_args, obj_kwargs))
    p.start()
    # wait until the process is finished or timeout is reached
    p.join(timeout=timeout)
    # terminate the process if it is still alive
    if p.is_alive():
        logger.info('Process timeout and is alive, terminate it')
        p.terminate()
        time.sleep(0.1)
        i = 0
        while p.is_alive():
            i += 1
            if i <= 10 or i % 100 == 0:
                logger.warning(f'Process is still alive, kill it ({i})')
            p.kill()
            time.sleep(0.1)
    # get the result
    try:
        result = queue.get(block=False)
    except Empty:
        result = None
    queue.close()
    result = _check_result(result)
    result['elapsed_time'] = time.time() - start_time
    return result


def run_obj_func(obj_func, obj_args, obj_kwargs, timeout=None):
    if timeout is None:
        result = run_without_time_limit(obj_func, obj_args, obj_kwargs)
    else:
        if timeout <= 0:
            timeout = None  # run by Process without timeout
        result = run_with_time_limit(obj_func, obj_args, obj_kwargs, timeout)
    return result


def process_src_data(his):
    default = his.observations[0].objectives[0]

    # 用默认配置的五倍弱来填inf
    fill_val = default * 5 if default > 0 else default / 5
    for obs in his.observations:
        if not np.isfinite(obs.objectives[0]):
            obs.objectives[0] = fill_val
            print("find inf objective in %s, fill %f" % (his.task_id, fill_val))

    return his

def check_args(args):
    assert args.opt in ['RS', 'GP', 'GPF', 'SMAC', 'TPE', 'BOHB', 'MFSE', 'FlexHB'], args.opt
    assert args.expert in ['RS', 'GP', 'GPF', 'SMAC', 'TPE', 'BOHB', 'MFSE', 'FlexHB', 'none'], args.expert
    assert args.compress in ['REMBO', 'ALEBO', 'BAxUS', 'Dropout-VS', 'MCTS-VS', 'Gr-VS', 'ours', 'Add-GP-UCB', 'PP-GP-UCB', 'RPP-GP-UCB', 'none'], args.compress
    assert args.warm_start in ['RS', 'GP', 'GPF', 'SMAC', 'TPE', 'BOHB', 'MFSE', 'FlexHB', 'none'], args.warm_start
    assert args.transfer in ['RS', 'GP', 'GPF', 'SMAC', 'TPE', 'BOHB', 'MFSE', 'FlexHB', 'none'], args.transfer
    assert args.task >= 0

def build_bo_optimizer(args, **kwargs):
    """构建BO优化器的辅助函数，用于RPP-GP-UCB等压缩策略"""
    from Optimizer.smbo import SMBO
    
    return SMBO(
        config_space=kwargs['config_space'], eval_func=kwargs['eval_func'],
        iter_num=args.iter_num, per_run_time_limit=kwargs.get('per_run_time_limit', None), 
        meta_feature=kwargs.get('meta_feature', None),
        source_hpo_data=kwargs.get('source_hpo_data', None), 
        ep_args=kwargs.get('ep_args', None), ep_strategy=getattr(args, 'expert', 'none'),
        method_id=getattr(args, 'opt', 'SMAC'), task_id=args.task, 
        target=kwargs['target'], task_str=kwargs.get('task_str', 'run'),
        cprs_strategy=kwargs.get('cprs_strategy', 'none'), 
        space_history=kwargs.get('source_compress_data', None), 
        cprs_k=getattr(args, 'cprs_k', 50),
        ws_strategy=getattr(args, 'warm_start', 'none'), 
        ws_args=kwargs.get('ws_args', None), 
        tl_strategy=getattr(args, 'transfer', 'none'), 
        tl_args=kwargs.get('tl_args', None),
        backup_flag=getattr(args, 'backup_flag', False), 
        seed=args.seed, rand_prob=getattr(args, 'rand_prob', 0.15), 
        rand_mode=getattr(args, 'rand_mode', 'ran'), 
        config_modifier=kwargs.get('config_modifier', None), 
        **kwargs
    )