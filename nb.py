import os

import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 指定使用GPU0

print(torch.cuda.is_available())

def process_data(filename, start_date=None, end_date=None):
    # 读取数据
    df = pd.read_csv(filename)

    # 将第一列设置为时间索引并转换为日期类型
    df['Unnamed: 0'] = pd.to_datetime(df['Unnamed: 0'], format='%Y%m')
    df.set_index('Unnamed: 0', inplace=True)

    # 如果提供了起止日期，则截取该时间段的数据
    if start_date and end_date:
        df = df.loc[pd.to_datetime(start_date, format='%Y%m'):pd.to_datetime(end_date, format='%Y%m')]

    # 读取Y数组（以xr_开头的变量）
    Y = df.filter(regex='^xr_').values

    # 读取F数组（列名以m结尾的数据）
    F = df.filter(regex='m$').values

    X = df.filter(regex='^f_').values

    # 读取M数组（其他变量）
    M = df.drop(columns=df.filter(regex='(^xr_)|(^f_)|m$').columns).values

    # 返回时间索引列
    time = df.index

    # Y = Y[:, 5].reshape((Y.shape[0]), 1)
    return Y * 100, X, F, M, time


def R2OOS(y_true, y_forecast):
    # Compute conidtional mean forecast
    y_condmean = np.divide(y_true.cumsum(), (np.arange(y_true.size) + 1))

    # lag by one period
    y_condmean = np.insert(y_condmean, 0, np.nan)
    y_condmean = y_condmean[:-1]
    y_condmean[np.isnan(y_forecast)] = np.nan

    # Sum of Squared Resids
    SSres = np.nansum(np.square(y_true - y_forecast))
    SStot = np.nansum(np.square(y_true - y_condmean))

    return 1 - SSres / SStot


# 全局变量定义
HIDDEN_LAYERS_MODEL_ONE = 1
HIDDEN_NODES_MODEL_ONE = [3]
OUTPUT_NODES_MODEL_ONE = 3

HIDDEN_LAYERS_MODEL_TWO = 1
HIDDEN_NODES_MODEL_TWO = [32]
OUTPUT_NODES_MODEL_TWO = 4

# 全局变量定义
MODEL_TWO_COUNT = 0  # 可以设置为任意非负整数，包括0

class Model_one(nn.Module):
    def __init__(self, input_size, hidden_sizes=[64], dropout_rate=0.5, output_nodes=OUTPUT_NODES_MODEL_ONE):
        super(Model_one, self).__init__()
        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(dropout_rate)

        # 定义隐藏层
        prev_size = input_size
        for hidden_size in hidden_sizes:
            self.layers.append(nn.Linear(prev_size, hidden_size))
            prev_size = hidden_size

        self.norm_layer = nn.BatchNorm1d(hidden_sizes[-1])  # 添加批量归一化层
        self.output_layer = nn.Linear(hidden_sizes[-1], output_nodes)

    def forward(self, x):
        for layer in self.layers:
            x = F.relu(layer(x))
            x = self.dropout(x)
        x = self.norm_layer(x)  # 应用批量归一化
        x = self.output_layer(x)
        return x


class Model_two(nn.Module):
    def __init__(self, input_size, hidden_layers=HIDDEN_LAYERS_MODEL_TWO, dropout_rate=0.5,
                 hidden_nodes=HIDDEN_NODES_MODEL_TWO, output_nodes=OUTPUT_NODES_MODEL_TWO):
        super(Model_two, self).__init__()
        self.layers = nn.ModuleList()

        # 第一个隐藏层
        self.layers.append(nn.Sequential(
            nn.Linear(input_size, hidden_nodes[0]),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        ))

        # 添加额外的隐藏层
        for i in range(1, hidden_layers):
            self.layers.append(nn.Sequential(
                nn.Linear(hidden_nodes[i - 1], hidden_nodes[i]),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ))

        self.norm_layer = nn.BatchNorm1d(hidden_nodes[-1])  # 添加批量归一化层
        self.output_layer = nn.Linear(hidden_nodes[-1], output_nodes)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.norm_layer(x)  # 应用批量归一化
        x = self.output_layer(x)
        return x





class IntegratedModel(nn.Module):
    def __init__(self, input_size_model_one, input_size_model_two, output_size, dropout_rate=0.5,
                 model_two_count=MODEL_TWO_COUNT):
        super(IntegratedModel, self).__init__()
        self.model_one = Model_one(input_size_model_one, dropout_rate=dropout_rate)

        # 根据model_two_count动态创建Model_two实例的列表
        self.models_two = nn.ModuleList(
            [Model_two(input_size_model_two, dropout_rate=dropout_rate) for _ in range(model_two_count)])

        # 调整最终输出层的输入尺寸
        final_input_size = OUTPUT_NODES_MODEL_ONE + OUTPUT_NODES_MODEL_TWO * model_two_count
        self.final_layer = nn.Linear(final_input_size, output_size)

    def forward(self, x1, x2):
        out_one = self.model_one(x1)

        # 如果有Model_two实例，处理它们的输出
        outs_two = [model(x2) for model in self.models_two]
        if outs_two:
            combined_out_two = torch.cat(outs_two, dim=1)
            combined_out = torch.cat((out_one, combined_out_two), dim=1)
        else:
            combined_out = out_one

        final_out = self.final_layer(combined_out)
        return final_out


def hyperparameter_search(F_train, M_train, Y_train, dropout_rates, l2_regs, model_two_count=MODEL_TWO_COUNT,
                          batch_size=32):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 指定使用GPU0
    best_val_loss = np.inf
    best_params = None
    print("参数寻优中，请稍后~")
    for dropout_rate in dropout_rates:
        for l2_reg in l2_regs:
            # 注意这里传递了 model_two_count 参数
            model = IntegratedModel(F_train.shape[1], M_train.shape[1], Y_train.shape[1], dropout_rate,
                                    model_two_count=model_two_count).to(device)
            optimizer = optim.AdamW(model.parameters(), weight_decay=l2_reg)
            loss_fn = nn.MSELoss()

            # 划分训练集和验证集
            F_train_sub, F_val, M_train_sub, M_val, Y_train_sub, Y_val = train_test_split(
                F_train.cpu().numpy(), M_train.cpu().numpy(), Y_train.cpu().numpy(), test_size=0.15, random_state=3407)

            # 将 NumPy 数组转换回 tensors 并移至GPU
            F_train_sub, F_val = torch.tensor(F_train_sub, dtype=torch.float32, device=device), torch.tensor(F_val,
                                                                                                             dtype=torch.float32,
                                                                                                             device=device)
            M_train_sub, M_val = torch.tensor(M_train_sub, dtype=torch.float32, device=device), torch.tensor(M_val,
                                                                                                             dtype=torch.float32,
                                                                                                             device=device)
            Y_train_sub, Y_val = torch.tensor(Y_train_sub, dtype=torch.float32, device=device), torch.tensor(Y_val,
                                                                                                             dtype=torch.float32,
                                                                                                             device=device)

            # 准备训练数据的批次
            train_dataset = torch.utils.data.TensorDataset(F_train_sub, M_train_sub, Y_train_sub)
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

            early_stop_counter = 0
            min_val_loss = np.inf

            for epoch in tqdm(range(10000), desc=f"Optimizing: Dropout {dropout_rate}, L2 {l2_reg}"):
                model.train()
                for F_batch, M_batch, Y_batch in train_loader:
                    optimizer.zero_grad()
                    outputs = model(F_batch, M_batch)
                    loss = loss_fn(outputs, Y_batch)
                    loss.backward()
                    optimizer.step()

                model.eval()
                with torch.no_grad():
                    val_outputs = model(F_val, M_val)
                    val_loss = loss_fn(val_outputs, Y_val)

                if val_loss.item() < min_val_loss:
                    min_val_loss = val_loss.item()
                    early_stop_counter = 0
                else:
                    early_stop_counter += 1

                if early_stop_counter > 500:
                    print(f"\n早停，以防过拟合。min_val_loss:{min_val_loss}")
                    break

            if min_val_loss < best_val_loss:
                best_val_loss = min_val_loss
                best_params = {'dropout_rate': dropout_rate, 'l2_reg': l2_reg}

    print(f"\n最佳超参数：{best_params}")
    return best_params


def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def save_model_config(model, file_path):
    with open(file_path, 'w') as f:
        f.write(str(model))

def main(filename, start_date, end_date, split_date, model_two_count=MODEL_TWO_COUNT, n_trials=20, k_best=2):
    # 确保基础目录存在
    base_dir = './simplemodel1'
    results_dir = './simplemodel1/results'
    ensure_dir(base_dir)
    ensure_dir(results_dir)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 指定使用GPU0
    Y, F, M, X, time = process_data(filename, start_date, end_date)

    # 转换为Tensor并移至GPU
    Y_tensor = torch.FloatTensor(Y).to(device)
    F_tensor = torch.FloatTensor(F).to(device)
    M_tensor = torch.FloatTensor(M).to(device)

    # 数据索引拆分
    split_idx = np.where(time == pd.to_datetime(split_date, format='%Y%m'))[0][0]
    F_train, F_test = F_tensor[:split_idx], F_tensor[split_idx:]
    M_train, M_test = M_tensor[:split_idx], M_tensor[split_idx:]
    Y_train, Y_test = Y_tensor[:split_idx], Y_tensor[split_idx:]

    # 特征缩放
    scaler_F = MinMaxScaler()
    F_train = torch.FloatTensor(scaler_F.fit_transform(F_train.cpu().numpy())).to(device)
    F_test = torch.FloatTensor(scaler_F.transform(F_test.cpu().numpy())).to(device)

    scaler_M = MinMaxScaler()
    M_train = torch.FloatTensor(scaler_M.fit_transform(M_train.cpu().numpy())).to(device)
    M_test = torch.FloatTensor(scaler_M.transform(M_test.cpu().numpy())).to(device)

    # 初始化Ypre数组，维度与Y相同，用NaN填充
    Ypre = np.full(Y.shape, np.nan)

    predictions = []

    for i in tqdm(range(len(F_test)), desc="Processing Test Samples"):
        # 为每个时点创建子目录
        time_point_dir = os.path.join(base_dir, f'time_point_{i}')
        ensure_dir(time_point_dir)
        # 每隔48条数据重新进行超参寻优
        if i % 48 == 0:
            # 重新执行超参数寻优
            dropout_rates = [0.1,0.3,0.5]
            l2_regs = [0.01,0.04, 0.07, 0.001]
            best_params = hyperparameter_search(F_train, M_train, Y_train, dropout_rates, l2_regs,
                                                model_two_count=model_two_count, batch_size=32)
        trial_models = []
        for trial in range(n_trials):
            # 传入model_two_count到IntegratedModel
            model = IntegratedModel(F_train.shape[1], M_train.shape[1], Y_train.shape[1], best_params['dropout_rate'],
                                    model_two_count=model_two_count).to(device)
            optimizer = optim.AdamW(model.parameters(), weight_decay=best_params['l2_reg'])
            loss_fn = nn.MSELoss()

            F_train_val, F_val, M_train_val, M_val, Y_train_val, Y_val = train_test_split(
                F_train.cpu().numpy(), M_train.cpu().numpy(), Y_train.cpu().numpy(), test_size=0.15, random_state=3407)
            F_train_val, F_val = torch.tensor(F_train_val, dtype=torch.float32, device=device), torch.tensor(F_val,
                                                                                                             dtype=torch.float32,
                                                                                                             device=device)
            M_train_val, M_val = torch.tensor(M_train_val, dtype=torch.float32, device=device), torch.tensor(M_val,
                                                                                                             dtype=torch.float32,
                                                                                                             device=device)
            Y_train_val, Y_val = torch.tensor(Y_train_val, dtype=torch.float32, device=device), torch.tensor(Y_val,
                                                                                                             dtype=torch.float32,
                                                                                                             device=device)

            train_dataset = torch.utils.data.TensorDataset(F_train_val, M_train_val, Y_train_val)
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)

            min_val_loss = np.inf
            early_stop_counter = 0
            for epoch in range(10000):
                model.train()
                for F_batch, M_batch, Y_batch in train_loader:
                    optimizer.zero_grad()
                    outputs = model(F_batch, M_batch)
                    loss = loss_fn(outputs, Y_batch)
                    loss.backward()
                    optimizer.step()

                model.eval()
                with torch.no_grad():
                    val_outputs = model(F_val, M_val)
                    val_loss = loss_fn(val_outputs, Y_val)

                if val_loss.item() < min_val_loss:
                    min_val_loss = val_loss.item()
                    early_stop_counter = 0
                else:
                    early_stop_counter += 1

                if early_stop_counter > 500:
                    print(f"Early stopping at trial {trial + 1}, epoch {epoch + 1} with val loss: {min_val_loss}")
                    break

            trial_models.append((min_val_loss, model))

        trial_models.sort(key=lambda x: x[0])
        best_models = trial_models[:k_best]

        model_preds = []
        for _, model in best_models:
            model.eval()
            with torch.no_grad():
                test_pred = model(F_test[i:i + 1], M_test[i:i + 1])
                model_preds.append(test_pred.cpu().numpy())

        # 对于每个时间点的预测值，填充Ypre数组
        avg_pred = np.mean(model_preds, axis=0)
        predictions.append(avg_pred.flatten().tolist())

        # 更新Ypre数组对应的位置
        Ypre[split_idx + i] = avg_pred

        # 保存网络配置
        config_path = os.path.join(time_point_dir, 'network_config.txt')
        save_model_config(model, config_path)

        # 保存最佳模型
        for rank, (loss, model) in enumerate(best_models):
            model_path = os.path.join(time_point_dir, f'best_model_{rank+1}.pt')
            torch.save(model.state_dict(), model_path)

    # 保存结果
    predictions = np.array(predictions)
    actuals = Y_test.cpu().numpy()
    results_df = pd.DataFrame({'Predictions': predictions.flatten(), 'Actuals': actuals.flatten()})
    results_df_path = os.path.join(results_dir, 'simpleresults1.csv')
    results_df.to_csv(results_df_path, index=False)
    print("\n预测完成，结果已保存至", results_df_path)

    R2OOS_scores_per_dimension = [R2OOS(Y[:, i], Ypre[:, i]) for i in range(predictions.shape[1])]
    print("R2OOS Scores Per Dimension:", R2OOS_scores_per_dimension)


main('./data.csv', '197108', '202112', '199001')
