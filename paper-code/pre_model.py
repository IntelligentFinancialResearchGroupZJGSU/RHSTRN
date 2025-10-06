import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from Graph import GraphConv
from data_packing import data_loder
import math
import config

train_file = config.datafile_source+'/train_data.csv'
test_file = config.datafile_source+'/test.csv'
cosine_train = config.datafile_source+'/cosine_train.csv'
cosine_test = config.datafile_source+'/cosine_test.csv'


class HRSE_pre(nn.Module):
    def __init__(self, input_size, hidden_dim, hidden_dim2):
        super(HRSE_pre, self).__init__()
        self.GraphConv1 = GraphConv(input_size, hidden_dim)
        self.GraphConv2 = GraphConv(hidden_dim, hidden_dim2)
        self.GraphConv3 = GraphConv(hidden_dim2, 2)
        # self.fc = nn.Linear(32, 2)

    def forward(self, datalist):
        for data in datalist:
            x, edge_index, = data.x, data.edge_index
            x1 = F.relu(self.GraphConv1(x, edge_index))
            x2 = self.GraphConv2(x1, edge_index)
            x3 = self.GraphConv3(x2, edge_index)
        out = x3
        out = F.softmax(out, dim=1)
        return out


def train_graph(model, device, train_loader, epoch, learning_rate):
    losses = []
    accuracies = []
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()
    model.train().to(device)
    for ep in range(epoch):
        train_loss = 0
        correct = 0
        total = 0
        for train_batch, label in train_loader:
            optimizer.zero_grad()
            output = model(train_batch).to(device)
            labels = label.to(device)
            loss = criterion(output, labels.squeeze())
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, predicted = torch.max(output, 1)  # 获取预测的类别
            correct += (predicted == labels.squeeze()).sum().item()  # 计算正确预测的数量
            total += labels.size(1)  # 更新总样本数
        losses.append(train_loss)
        accuracy = correct / total  # 计算精确度
        accuracies.append(accuracy)
        print(
            f'Epoch {ep + 1}/{ep}, Loss: {train_loss / len(train_loader):.4f}, Accuracy: {accuracy * 100 :.2f}%')
    torch.save(model.GraphConv1.state_dict(), 'SZ50E/GraphConv1.pth')
    torch.save(model.GraphConv2.state_dict(), 'SZ50E/GraphConv2.pth')


def evaluate_graph(model, device, test_loader):
        model.eval().to(device)
        correct = 0
        total = 0
        predicate_out = []
        for test_batch, label in test_loader:
            with torch.no_grad():
                output = model(test_batch).to(device)
                predicate_out.append(output)
                labels = label.to(device)
                _, predicted = torch.max(output, 1)  # 获取预测的类别
                correct += (predicted == labels.squeeze()).sum().item()  # 计算正确预测的数量
                total += labels.size(1)  # 更新总样本数
        accuracy = correct / total  # 计算精确度
        predicate_out = torch.stack(predicate_out, dim=0)
        print(f'Accuracy: {accuracy*100 :.2f}%')
        return predicate_out