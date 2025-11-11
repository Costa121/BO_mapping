from .base import *
import numpy as np
import copy
from sklearn.preprocessing import MinMaxScaler

class EuclideanMapper(BaseMapper):
    def __init__(self):
        super().__init__()

        self.map_strategy = "euclidean"
        self.scaler = MinMaxScaler()

    """
    先归一化，再计算任务环境向量之间的欧几里得距离
    """
    @staticmethod
    def context_distance(context1, context2):
        # context1 = np.array(copy.deepcopy(context1), dtype=float)
        # context2 = np.array(copy.deepcopy(context2), dtype=float)
        # for i in range(len(context1)):
        #     if not max_context[i] == min_context[i]:
        #         context2[i] = (context2[i] - min_context[i]) / (max_context[i] - min_context[i])
        #         context1[i] = (context1[i] - min_context[i]) / (max_context[i] - min_context[i])
        #
        # context1 = np.array(context1)
        # context2 = np.array(context2)

        dist = np.linalg.norm(context1 - context2)
        return dist

    # 返回相似性降序排列编号，以及对应的相似度
    def map(self, target_history: History, source_hpo_data: list[History]) -> list[Tuple[int, float]]:

        target_context = target_history.meta_info['meta_feature']
        source_contexts = source_hpo_data[0].meta_info['meta_feature'].copy()
        for i in range(1, len(source_hpo_data)):
            source_contexts = np.vstack((source_contexts, source_hpo_data[i].meta_info['meta_feature'].copy()))

        self.scaler.fit(source_contexts)
        source_contexts = self.scaler.transform(source_contexts)
        target_context = self.scaler.transform([target_context])[0]

        source_contexts[np.isnan(source_contexts)] = 0
        target_context[np.isnan(target_context)] = 0

        dists = []
        for i, source_context in enumerate(source_contexts):
            dist = self.context_distance(target_context, source_context)
            dists.append((i, -dist))

        dists = sorted(dists, key=lambda x: x[1], reverse=True)

        return dists


class CosineMapper(BaseMapper):
    def __init__(self):
        super().__init__()

        self.map_strategy = "cosine"

    def context_distance(self, context1, context2):
        context1 = np.array(context1)
        context2 = np.array(context2)

        context1[np.isnan(context1)] = 0
        context2[np.isnan(context2)] = 0

        dist = np.dot(context1, context2) / (np.linalg.norm(context1) * np.linalg.norm(context2))
        return dist

    def map(self, target_history: History, source_hpo_data: list[History]) -> list[Tuple[int, float]]:

        target_context = target_history.meta_info['meta_feature']

        dists = []
        for i, hpo_data in enumerate(source_hpo_data):
            dist = self.context_distance(target_context, hpo_data.meta_info['meta_feature'])
            dists.append((i, dist))

        dists = sorted(dists, key=lambda x: x[1], reverse=True)

        return dists
