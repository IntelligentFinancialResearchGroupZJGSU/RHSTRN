import pandas as pd
import numpy as np
import torch
from scipy.stats import norm
from sklearn.metrics.pairwise import cosine_similarity


def split_data(graph_data_list, window_size):
    input_sequences = []
    if window_size == 1:
        for t in range(0, len(graph_data_list)):
            input_seq = [graph_data_list[t]]
            input_sequences.append(input_seq)
    else:
        for t in range(window_size, len(graph_data_list)):
            input_seq = graph_data_list[t-window_size:t]
            input_sequences.append(input_seq)
    return input_sequences


def build_edge(datafile, threshold, window):  # 计算余弦相似度选取边
    edge_list = []
    edges_all = []
    alpha = 0.95
    data = pd.read_csv(datafile)
    data['time'] = pd.to_datetime(data['time'])
    grouped = data.groupby('time')
    time_point_data = {}
    for time, group in grouped:  # 数据打包成元组，并去除冗余列
        feature_cols = [col for col in group.columns if col not in ['time', 'thscode']]
        time_point_data[time] = tuple(map(tuple, group.loc[:, feature_cols].values))
    for time, data in time_point_data.items():
        data_x = tuple(tuple(x) for x in data)
        x = torch.tensor(data_x, dtype=torch.float32).contiguous()
        edge_list.append(x)
    edge_list = split_data(edge_list, window)
    for edge_window in edge_list:
        z_alpha = (1 - alpha ** window) / (1 - alpha)
        edg_index = set()
        n = edge_window[0].shape[0]
        min_edges = max(int(n * 1.5), 20)
        cosine_sum = np.zeros((n, n), dtype=np.float32)
        for day, edge_day in enumerate(edge_window):
            w = alpha**(window - day - 1)
            cosine = cosine_similarity(edge_day)
            cosine_sum += w*cosine
        cosine_sum /= z_alpha
        mu = cosine_sum.mean()
        sigma = cosine_sum.std()
        z_thresh = norm.ppf(1 - threshold / 100)
        thresholds = np.percentile(cosine_sum, 100 - threshold)
        for i in range(n):
            for j in range(n):
                if i != j:
                    z = (cosine_sum[i, j] - mu) / sigma
                    if z > z_thresh:
                        edg_index.add((i, j))
        if len(edg_index) < min_edges:
            upper_triangle_indices = [(i, j) for i in range(n) for j in range(n) if i != j]
            flat_sims = [cosine_sum[i, j] for i, j in upper_triangle_indices]
            sorted_indices = np.argsort(flat_sims)[::-1]  # 从大到小排序
            for idx in sorted_indices:
                i, j = upper_triangle_indices[idx]
                edg_index.add((i, j))
                if len(edg_index) >= min_edges:
                    break
        edges_all.append(list(edg_index))
    return edges_all


