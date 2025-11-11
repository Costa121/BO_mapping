# License: MIT
import argparse
import json
import time
from build_utils import build_optimizer
from openbox import logger
import pickle
import os
from ConfigSpace import Configuration, ConfigurationSpace
import numpy as np
from openbox.utils.history import History
from openbox import space as sp
from test_functions import Branin, Hartmann6, Gramacy

parser = argparse.ArgumentParser()
parser.add_argument('--opt', type=str, default='SMAC',
                    choices=['RS', 'contextBO', 'GP', 'GPF', 'SMAC', 'TPE', 'BOHB_GP', 'BOHB_SMAC', 'MFSE_GP', 'MFSE_SMAC', 'TAS-BO'])
parser.add_argument('--compress', type=str, default='none',
                    choices=['none', 'REMBO', 'ALEBO', 'BAxUS', 'Dropout-VS', 'MCTS-VS', 'Gr-VS', 'ours', 'Add-GP-UCB', 'PP-GP-UCB','RPP-GP-UCB'])
parser.add_argument('--cprs_k', type=int, default=4)

parser.add_argument('--target', type=str, default='test_complete')
parser.add_argument('--task', type=str, default='test')
parser.add_argument('--task_str', type=str, default='')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--rand_prob', type=float, default=0.15)
parser.add_argument('--rand_mode', type=str, default='rs', choices=['ran', 'rs'])
parser.add_argument('--iter_num', type=int, default=100)

parser.add_argument('--expert', type=str, default='none')
parser.add_argument('--warm_start', type=str, default='none')
parser.add_argument('--transfer', type=str, default='none')
parser.add_argument('--backup_flag', type=bool, default=False)
parser.add_argument('--safe_flag', type=bool, default=False)

parser.add_argument('--func', type=str, default='Branin', choices=['Branin', 'Hartmann6', 'Gramacy'])
parser.add_argument('--dim', type=int, default=100)

args = parser.parse_args()

target = args.target

if args.func == 'Branin':
    executor = Branin(2, 100)
elif args.func == 'Hartmann6':
    executor = Hartmann6(6, 100)
elif args.func == 'Gramacy':
    executor = Gramacy(2, 100)
    
source_hpo_data = None
source_compress_data = None

def get_space(d):
    space = sp.Space()
    tot_len = len(str(d))
    for i in range(d):
        name = 'x' + '0' * (tot_len - len(str(i))) + str(i)
        space.add_variable(sp.Real(name, lower=-1.0, upper=1.0, default_value=0))
    space.seed(args.seed)
    return space
space = get_space(args.dim)

ws_args = {
    'init_num': 5,
    'topk': 3,
    'inner_surrogate_model': 'prf'
}

opt_kwargs = {
    'config_space': space,
    'eval_func': executor,
    'target': target,
    'task_str': args.task_str,
    'meta_feature': {'meta_feature': None, 'ini_context': None},
    'ws_args': ws_args
}
optimizer = build_optimizer(args, **opt_kwargs)

for i in range(optimizer.iter_num):
    optimizer.run_one_iter()
