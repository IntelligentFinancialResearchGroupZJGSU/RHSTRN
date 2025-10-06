import torch
from torch.utils.data import Dataset, TensorDataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import numpy as np
from build_edg import build_edge, split_data
import pandas as pd
import config
import itertools

device = config.device


class GraphDataset(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]


# 打包数据成图数据
def packing(datafile, cosine_file, threshold, window):
    graph_data_list = []
    x_all = []
    y_all = []
    num = 0
    edg_index = build_edge(cosine_file, threshold, window)
    data = pd.read_csv(datafile, parse_dates=['time'])
    data['time'] = pd.to_datetime(data['time'])
    grouped = data.groupby('time')
    time_point_data = {}
    for time, group in grouped:  # 数据打包成元组，并去除冗余列
        time_point_data[time] = tuple(map(tuple, group.drop(columns=['time', 'thscode']).values))
    for time, data in time_point_data.items():
        data_x = tuple(tuple(x[:-1]) for x in data)  # 取出每一条数据的前四列，格式：o+h+l+c +volume+amount
        data_y = tuple((x[-1]) for x in data)  # 取出趋势标签
        data_y = np.array(data_y, dtype=np.int64)
        x = torch.tensor(data_x, dtype=torch.float32).contiguous().to(device)
        y = torch.tensor(data_y, dtype=torch.int64).t().contiguous().squeeze().to(device)
        x_all.append(x)
        y_all.append(y)
    for edg in edg_index:
        edge = torch.tensor(edg).t().contiguous().to(device)
        graph_list = []
        for x, y in zip(x_all[num:num+window], y_all[num:num+window]):
            graph = Data(x=x, edge_index=edge, y=y)
            graph_list.append(graph)
        graph_data_list.append(graph_list)
        num += 1
    return graph_data_list, y_all


#  按天数划分，打包数据dataloader
def data_loder(datafile, cosine_file, threshold, window_size):
    graph_data_list, labels = packing(datafile, cosine_file, threshold, window_size)
    if window_size ==1: label = labels[window_size-1:len(labels)]
    else: label = labels[window_size:len(labels)]
    dataset = GraphDataset(data=graph_data_list, label=label)
    data_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False)
    for data_batch, label in data_loader:  # 数据放入gpu
        data_batch_batch = [x.to(device) for x in data_batch]
        labels = [y.to(device) for y in label]
    return data_loader

def shap_loder(datafile):
    data = pd.read_csv(datafile, parse_dates=['time'])
    data['time'] = pd.to_datetime(data['time'])
    grouped = data.groupby('thscode')
    stock_data_list = []
    for stock, group in grouped:  # 数据打包成元组，并去除冗余列
        stock_list = tuple(
                map(tuple, group.drop(columns=['time', 'thscode']).values))
        stock_data_list.append(stock_list)
    stock_data_list = np.array(stock_data_list)
    x = torch.tensor(stock_data_list[:, :, :-1], dtype=torch.float32)
    y = torch.tensor(stock_data_list[:, :, -1], dtype=torch.float32)
    return x, y
