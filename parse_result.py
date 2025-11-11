import pickle as pkl
import os
import json
import numpy as np
from matplotlib import pyplot as plt

path = './results/math_test/SMACrs/'
methods = ['none', 'REMBO', 'ALEBO', 'BAxUS', 'Dropout-VS', 'Gr-VS', 'ours']
workloads = ['Branin', 'Hartmann6', 'Gramacy']
res = []
for k in workloads:
    res.append([])
    for mth in methods:
        tmp = []
        files = os.listdir(path)
        for file in files:
            fwl = file.split('_')[0]
            fmth = file.split('_')[1]
            if fwl == k and fmth == mth:
                with open(path + file, 'r') as f:
                    data = json.load(f)
                obs = data.pop('observations')
                y = [ob['objectives'][0] for ob in obs]
                for i in range(1, len(y)):
                    y[i] = min(y[i-1], y[i])
                tmp.append(y)
        tmp = np.mean(tmp, axis = 0).tolist()
        res[-1].append(tmp)
res = np.array(res)
res[0] = res[0] - 0.397887
res[1] = res[1] + 3.32237
res[2] = res[2] + 0.428882

algos = ['BO', 'REMBO', 'ALEBO', 'BAxUS', 'Dropout-VS', 'Gr-VS', 'ours']
idx = [i + 1 for i in range(100)]
for i, k in enumerate(workloads):
    kres = {}
    kstd = {}
    for j, algo in enumerate(algos):
        kres[algo] = res[i][j]
        kstd[algo] = res[i][j]
    with open('./' + k + '.pkl', 'wb') as f:
        pkl.dump([algos, idx, kres, kstd], f)

rank = res.copy()
for i in range(len(workloads)):
    for j in range(100):
        rank[i, :, j][np.argsort(res[i, :, j])] = np.arange(len(methods)) + 1
# for i in range(len(workloads)):
#     for j in range(40):
#         for k in range(3):
#             rank[i, :, k, j][np.argsort(res[i, :, k, j])] = np.arange(len(methods)) + 1
# rank = np.mean(rank, axis = 2)

mean_rank = np.mean(rank, axis = 0).tolist()

res = {}
std = {}
for i, algo in enumerate(algos):
    res[algo] = mean_rank[i]
    std[algo] = mean_rank[i]
    with open('./rank.pkl', 'wb') as f:
        pkl.dump([algos, idx, res, std], f)