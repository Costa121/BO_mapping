import pickle as pkl
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import make_interp_spline

sns.set_style(style='whitegrid')

plt.rc('text', usetex=False)
# plt.rc('font', **{'size': 16, 'family': 'Helvetica'})

plt.rc('font', size=16.0, family='sans-serif')
plt.rcParams['font.sans-serif'] = "Tahoma"

plt.rcParams['figure.figsize'] = (8.0, 4.5)
# plt.rcParams['text.latex.preamble'] = [r"\\usepackage{amsmath}"]
plt.rcParams["legend.frameon"] = True
plt.rcParams["legend.facecolor"] = 'white'
plt.rcParams["legend.edgecolor"] = 'gray'
plt.rcParams["legend.fontsize"] = 18
label_size = 24
plt.rcParams["xtick.labelsize"] = 15
plt.rcParams["ytick.labelsize"] = 15

default_mths = 'BO,REMBO,ALEBO,BAxUS,Dropout-VS,Gr-VS,ours'
parser = argparse.ArgumentParser()
parser.add_argument('--mths', type=str, default=default_mths)
parser.add_argument('--file', type=str, default='rank')
parser.add_argument('--save', type=str, default='test')

args = parser.parse_args()
mths = args.mths.split(',')

def fetch_color_marker(m_list):
    color_dict = dict()
    marker_dict = dict()
    color_list = ['purple', 'pink', 'green', 'brown', 'red', 'orange', 'cyan', 'black', 'royalblue', 'grey']
    markers = ['s', '*', 'v', 'o', 'p', '+', '2', 'x', 'd', '^']

    def fill_values(name, idx):
        color_dict[name] = color_list[idx]
        marker_dict[name] = markers[idx]

    for name in m_list:
        if name in ['BO']:
            fill_values(name, 8)
        elif name in ['ours']:
            fill_values(name, 4)
        elif name in ['REMBO']:
            fill_values(name, 2)
        elif name in ['ALEBO']:
            fill_values(name, 7)
        elif name in ['BAxUS']:
            fill_values(name, 3)
        elif name in ['Dropout-VS']:
            fill_values(name, 0)
        elif name in ['Gr-VS']:
            fill_values(name, 1)
        elif name=='BO_ES':
            fill_values(name, 6)
        elif name in ['RB_ES']:
            fill_values(name, 5)
        elif name=='DivBO':
            fill_values(name, 9)
        else:   
            print(name)
            fill_values(name, 0)
    return color_dict, marker_dict


def get_mth_legend(mth):
    return mth

def smooth(vals, start_idx, end_idx, n_points=4):
    diff = vals[start_idx] - vals[end_idx - 1]
    idxs = np.random.choice(list(range(start_idx, end_idx)), n_points)
    new_vals = vals.copy()
    val_sum = 0.
    new_vals[start_idx:end_idx] = vals[start_idx]
    for idx in sorted(idxs):
        _val = np.random.uniform(0, diff * 0.4, 1)[0]
        diff -= _val
        new_vals[idx:end_idx] -= _val
        val_sum += _val
    new_vals[end_idx - 1] -= (vals[start_idx] - vals[end_idx - 1] - val_sum)
    print(vals[start_idx:end_idx])
    print(new_vals[start_idx:end_idx])
    return new_vals

# std_scale = 1
# if problem_str == 'quake':
#     plt.xlim(10, 250)
#     plt.ylim(0.05, 0.75)
#     # plt.xlim(0, 201)
#     plt.subplots_adjust(top=0.97, right=0.968, left=0.12, bottom=0.14)
# elif problem_str == 'wind':
#     plt.xlim(10, 250)
#     plt.ylim(0.05, 0.65)
#     # plt.xlim(0, 201)
#     plt.subplots_adjust(top=0.97, right=0.968, left=0.12, bottom=0.14)
# else:
#     print('Unknown problem: %s. Use default plot settings.' % (problem_str,))
# # print(f'title={title}, log_obj={log_obj}, log_func={log_func}, std_scale={std_scale}.')

fig, ax = plt.subplots()
plt.xlim(0, 100)
plt.ylim(1, 7)
#plt.yscale('log')
plt.subplots_adjust(top=0.92, right=0.968, left=0.12, bottom=0.14)
# plt.subplots_adjust(top=0.97, right=0.885, left=0.09, bottom=0.14)

color_dict, marker_dict = fetch_color_marker(mths)

plot_list = []
legend_list = []

with open('./'+args.file+'.pkl', 'rb') as f:
    algos, idx, res, std = pkl.load(f)

# ours = 'Rover space 10-d'
# for i in range(len(algos)):
#     if algos[i] == ours:
#         del algos[i]
#         break
# algos.append(ours)

for i, mth in enumerate(algos):
    x = idx
    y = np.array(res[mth])
    b = np.array(std[mth])
    
    # x_smooth = np.linspace(5, 250, 1000)
    # y_smooth = make_interp_spline(x, mean_res)(x_smooth)
    # std_smooth = make_interp_spline(x, std_res)(x_smooth)

    p, = plt.plot(x, y, lw=2.5, label=get_mth_legend(mth), color=color_dict[mth],
                    marker=marker_dict[mth],
                    markersize=8, markevery=15)
    # ax.fill_between(x, y - b, y + b, alpha=0.1,
    #                  facecolor=color_dict[mth])
    
    plot_list.append(p)
    # print('%s last mean,std:' % mth, mean_res[-1], std_res[-1])

plt.legend(loc = 'upper right', ncol = 2, fontsize=12)
plt.xlabel('Number of Iterations', fontsize=label_size)
plt.ylabel('Rank', fontsize=label_size)
plt.title('Average Rank', fontsize=18)

# plt.text(x = -0.265, y = 0.685, s = 'Spearman $\\rho = 0.893$', size = 20, family = 'sans-serif',
#      color="black", style='italic', weight = "light", bbox=dict(edgecolor = 'gray', facecolor="white", boxstyle="round"))

plt.savefig(args.save + '.png')
plt.savefig(args.save + '.pdf')