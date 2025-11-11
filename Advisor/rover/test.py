import numpy as np
from train_model import calculate_similarity, train_model
from transfer import get_transfer_suggestion, get_transfer_tasks
from test_function import Hartmann_3

# 随机搜索，根据meta_feature构建数学函数并生成iters轮的观察历史
def get_history(meta, iters = 50):
    meta = meta * 0.2 - 0.1
    meta[0] += 1
    test_func = Hartmann_3(scale = meta[0], trans = meta[1], sft = meta[2:])
    his = []
    for i in range(iters):
        conf = np.random.rand(3)
        his.append([conf, test_func(conf)])
    return his

if __name__ == '__main__':
    # 以下是训练模型部分的示例代码
    n_sim = 50

    # 随机生成用于训练模型的任务的meta_feature
    sim_meta_features = np.random.rand(n_sim, 5)
    # 用随机搜索的方式生成用于训练模型的任务的观察历史
    sim_his = [get_history(sim_meta_features[i], 25) for i in range(n_sim)]

    # 计算相似度矩阵
    sim = calculate_similarity(sim_his)
    print(sim[0])

    # 训练模型并保存
    train_model(sim_meta_features, sim)


    # 以下是获取迁移学习推荐配置部分的示例代码
    n_src = 100

    # 随机生成备选的用于迁移的源任务的meta_feature
    src_meta_features = np.random.rand(n_src, 5)
    # 用随机搜索的方式生成备选的用于迁移的源任务的观察历史
    src_his = [get_history(src_meta_features[i], 25) for i in range(n_src)]
    # 随机生成目标任务的meta_feature
    target_meta_feature = np.random.rand(5)
    # 随机搜索生成目标任务的前3轮观察历史
    target_his = get_history(target_meta_feature, 3)

    # 根据meta_featrue生成目标函数
    tmp = target_meta_feature * 0.2 - 1
    tmp[0] += 1
    target_func = Hartmann_3(scale = tmp[0], trans = tmp[1], sft = tmp[2:])

    # 首先对备选任务进行筛选，获取用于迁移任务的index
    idx = get_transfer_tasks(src_meta_features, target_meta_feature, num = 5, theta = 0.7)
    print(idx)

    # 只有有任务可迁移才进行迁移
    if len(idx) > 0:
        # 模拟获取被选出的任务的历史
        transfer_his = []
        for id in idx:
            transfer_his.append(src_his[id])
        
        # 模拟进行3轮迁移学习
        for i in range(3):
            # 调用函数获取迁移学习的推荐配置
            config = get_transfer_suggestion(transfer_his, target_his)
            target_his.append([config, target_func(config)])

        for conf, obj in target_his:
            print(conf, obj)
    else:
        print('No suitable source task.')