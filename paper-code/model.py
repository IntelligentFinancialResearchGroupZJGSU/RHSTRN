import os
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from Graph import GraphConv
from data_packing import data_loder, shap_loder
from sklearn.metrics import f1_score,  roc_auc_score
import matplotlib.pyplot as plt
import numpy as np

train_file = config.datafile_source+'/train_data.csv'
val_file = config.datafile_source+'/val_data.csv'
test_file = config.datafile_source+'/test.csv'
cosine_train = config.datafile_source+'/cosine_train.csv'
cosine_test = config.datafile_source+'/cosine_test.csv'
device = config.device


class HRSE(nn.Module):
    def __init__(self, input_size, hidden_dim1, hidden_dim2, lstm_hidden_size1, num_classes):
        super(HRSE, self).__init__()
        self.GraphConv1 = GraphConv(input_size, hidden_dim1)
        self.GraphConv2 = GraphConv(hidden_dim1, hidden_dim2)
        self.GraphConv1.load_state_dict(torch.load('SZ50E/GraphConv1.pth', map_location=config.device))
        self.GraphConv2.load_state_dict(torch.load('SZ50E/GraphConv2.pth', map_location=config.device))
        for param in self.GraphConv1.parameters():
            param.requires_grad = False
        for param in self.GraphConv2.parameters():
            param.requires_grad = False
        self.fc1 = nn.Linear(input_size, hidden_dim2)
        self.lstm1 = nn.LSTM(hidden_dim2, lstm_hidden_size1, batch_first=False, num_layers=1, bidirectional=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(lstm_hidden_size1*2, num_classes)

    def forward(self, graph_data_list):
        batch_outputs = []
        for data in graph_data_list:
            x, edge_index = data.x, data.edge_index
            embedding_self = F.relu(self.fc1(x))
            embedding_cos = F.relu(self.graphsage1(x, edge_index))
            embedding = self.graphsage2(embedding_cos, edge_index)+embedding_self
            batch_outputs.append(embedding.unsqueeze(0))
        graph_embeddings = torch.cat(batch_outputs, dim=0)
        graph_embeddings = graph_embeddings
        lstm_out, (cn,hn) = self.lstm1(graph_embeddings)
        out = F.elu(lstm_out)
        drop_out = self.dropout(out)
        out = drop_out[-1, :, :]
        stock_out = F.softmax(self.fc(out), dim=1)
        return stock_out


def train(model, device, val_loader, epoch, learning_rate, val_lstm):
    # accumulation_steps = 4
    best_loss = 0
    patience = 3
    num_bad_epochs = 0
    losses = []
    accuracies = []
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    for epoch in range(epoch):
        model.train().to(device)
        train_loss = 0
        correct = 0
        total = 0
        all_labels = []
        all_out = []
        all_predictions = []
        accumulation_steps = 4
        for i, (train_batch, label) in enumerate(val_loader):
            output = model(train_batch).to(device)
            labels = label.to(device)
            loss = criterion(output, labels.squeeze())
            # loss = loss / accumulation_steps  # 梯度缩放
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            # 每累计一定步数再执行一次优化器更新
            # if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
            #     optimizer.step()
            #     optimizer.zero_grad()
            train_loss += loss.item()
            _, predicted = torch.max(output, 1)  # 获取预测的类别
            correct += (predicted == labels.squeeze()).sum().item()  # 计算正确预测的数量
            total += labels.size(1)  # 更新总样本数
            all_labels.extend(labels.squeeze().cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
        accuracy = correct / total  # 计算精确度
        f1 = f1_score(all_labels, all_predictions, average="micro")  # F1 分数
        auc = roc_auc_score(all_labels, all_predictions)
        accuracies.append(accuracy)
        print(
            f'Epoch {epoch + 1}/{epoch}, Loss: {train_loss / len(val_loader):.4f}, '
            f'Accuracy: {accuracy * 100:.2f}%, F1 Score: {f1:.4f}'
            f" AUC: {auc:.4f}"
        )

        # 验证
        model.eval().to(device)
        totals = 0
        val_loss = 0
        corrects = 0
        with torch.no_grad():
            for val_batch, labels in val_lstm:
                val_outputs = model(val_batch).to(device)
                val_losses = criterion(val_outputs, labels.squeeze())
                val_loss += val_losses.item()
                _, predicted = torch.max(val_outputs, 1)  # 获取预测的类别
                corrects += (predicted == labels.squeeze()).sum().item()
                totals += labels.size(1)
            accuracys = corrects / totals
            val_loss = val_loss
            print(f"val_loss={val_loss:.4f}")
            print(accuracys)
        if accuracys >= best_loss:
            best_loss = accuracys
            num_bad_epochs = 0  # 重置计数
        else:
            num_bad_epochs += 1
        if num_bad_epochs > patience:
            print("早停触发！")
            break
    return losses, accuracies


def evaluate(model, device, test_loader):
    model.eval().to(device)
    predicate_out = []
    correct = 0
    total = 0
    all_labels = []
    all_predictions = []
    for test_batch, label in test_loader:
        with torch.no_grad():
            output = model(test_batch).to(device)
            predicate_out.append(output)
            labels = label.to(device)
            _, predicted = torch.max(output, 1)
            correct += (predicted == labels.squeeze()).sum().item()
            total += labels.size(1)  # 更新总样本数
            all_labels.extend(labels.squeeze().cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
    accuracy = correct / total  # 计算精确度
    auc = roc_auc_score(all_labels, all_predictions)
    f1_scores = f1_score(all_labels, all_predictions, average="micro")
    print(f'Accuracy: {accuracy*100 :.2f}%')
    predicate_out = torch.stack(predicate_out, dim=0)
    return predicate_out, f1_scores, auc


def assessment(predicate_out, device =config.device):
    initial_investment = torch.tensor(10000)
    result = (predicate_out[:, :, 0] < predicate_out[:, :, 1]).long()
    result = result.unsqueeze(-1).to(device)
    profit_data, _ = shap_loder(config.datafile_source + '/profit.csv')
    profit_data = profit_data[:, :, 3]
    profit_data = profit_data.unsqueeze(-1)
    profit_data = (profit_data[:, 1:, :] / profit_data[:, :-1, :])
    profit_data = profit_data[:, 19:, :]
    profit_data = profit_data.transpose(0, 1).to(device)
    benchmark_daily_returns = profit_data[194:, :, :].mean(dim=1).squeeze(-1).cpu().numpy()
    profit = profit_data * result
    count = (result == 1).sum(dim=1)
    profit = profit[193:, :, :]
    count = count[193:, :]
    total_assets = initial_investment
    total_asset_list = []
    for day in range(profit.shape[0]):
        if (count[day, :] == 0):
            total_assets = total_assets * 1
        else:
            total_assets = (total_assets / count[day, :]) * (profit[day, :, :].sum())
        total_asset_list.append(total_assets.item() if torch.is_tensor(total_assets) else total_assets)
        # with open('profit_sz50_lstm.txt', 'a') as f:
        #     f.write(str(total_assets.item() if torch.is_tensor(total_assets) else total_assets) + '\n')
    total_asset_array = np.array([
        x.cpu().item() if torch.is_tensor(x) else x
        for x in total_asset_list
    ])
    # 计算每日收益率（这里用对数收益率更稳健）
    daily_returns = np.diff(total_asset_array) / total_asset_array[:-1]
    # daily_returns = np.diff(np.log(total_asset_array))
    r_f_daily = (1 + 0.021) ** (1 / 252) - 1
    sharpe_ratio = np.sqrt(252) * ((np.mean(daily_returns) - r_f_daily) / np.std(daily_returns))

    # 最大回撤计算
    cumulative_max = np.maximum.accumulate(total_asset_array)
    drawdown = (cumulative_max - total_asset_array) / cumulative_max
    max_drawdown = np.max(drawdown)

    # 年化收益率估算
    total_period = len(total_asset_array) / 252
    annual_return = (total_asset_array[-1] / total_asset_array[0]) ** (1 / total_period) - 1

    # 卡尔马比率
    calmar_ratio = annual_return / max_drawdown if max_drawdown != 0 else np.nan

    days = list(range(len(daily_returns)))
    return {
        "final_asset": total_assets,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown,
        "calmar_ratio": calmar_ratio,
        "total_asset_list": total_asset_list
    }


if __name__ == '__main__':
    device = config.device
    node_feature = 32  # 特征
    hidden_dim1 = 32  # 嵌入维度
    hidden_dim2 = 16
    lstm_hidden_size1 = 48
    num_classes = 2  # 类别
    epoch = 100
    threshold = 10
    window = 20
    accuracies = []
    acu = []
    train_lstm = data_loder(train_file, cosine_train, threshold, window)
    test_lstm = data_loder(test_file, cosine_test, threshold, window)
    val_lstm = data_loder(val_file, cosine_test, threshold, window)
    model_lstm = HRSE(node_feature, hidden_dim1, hidden_dim2, lstm_hidden_size1, num_classes)
    loss_history, accuracy_history = train(model_lstm, device, train_lstm, epoch, learning_rate=0.001, val_lstm=val_lstm)
    out, accuracy, auc = evaluate(model_lstm, device, test_lstm)
    results = assessment(out)
    print(f"最终资产：{results['final_asset'].item():.2f}")
    print(f"夏普比率：{results['sharpe_ratio']:.4f}")
    print(f"最大回撤：{results['max_drawdown']:.4f}")
    print(f"卡尔马比率：{results['calmar_ratio']:.4f}")





