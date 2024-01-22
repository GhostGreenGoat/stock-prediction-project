import torch
from torch.nn import Module, LSTM, Linear
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import torch.nn as nn

from torch.utils.data import Dataset
torch.manual_seed(42)
from scipy.stats import spearmanr
import matplotlib.pyplot as plt

class CustomDataset(Dataset):
    def __init__(self,X,Y=None,index=None):
        # 初始化函数，在这里设置数据集的初始参数
        self.X = X
        self.Y = Y
        self.index = index
        self.len = len(np.unique(self.index[:,:,0]))
        self.section_data_indices = self.get_section_data(self.index)
        self.data_list = list(np.unique(self.index[:,:,0]))

    def __len__(self):
        # 返回数据集中的样本数
        return self.len

    def __getitem__(self, index):
        # 生成单个样本
        # 单个样本即为截面数据
        date = self.data_list[index]
        sample_indices = self.section_data_indices[date] # 该日所有股票在self.index中的索引
        sample_index = self.index[sample_indices] #该日所有股票在原始df中的索引 形状为（section_size,timestep,2）
        if sample_index.shape[0] != 0:
            last_date = sample_index[0,1,0] #该日最后一个时间步的日期
        else:
            last_date = 0
        # 将索引转换为元组列表
        index_tuples = [tuple(idx) for day_indices in sample_index for idx in day_indices]

        # 一次性提取所有数据
        sample_X = self.X.loc[index_tuples]
        # 重塑结果以匹配所需形状
        sample_X = sample_X.values.reshape(sample_index.shape[0], sample_index.shape[1], self.X.shape[1])
        if not isinstance(self.Y,type(None)):
            sample_Y = self.Y.loc[index_tuples]
            sample_Y = sample_Y.values.reshape(sample_index.shape[0], sample_index.shape[1], 1)
            return (sample_X,sample_Y)
        else:
            return sample_X,last_date
    
    def get_section_data(self,data):
        dates = self.index[:,:,0]

        # 获取所有唯一的日期
        unique_dates = np.unique(dates)

        # 为每个日期创建一个字典来保存数据
        date_groups ={date: [] for date in unique_dates}

        # 对每个日期进行分组
        for i, date in enumerate(dates[:,0]):  # 假设每个时间步的日期都相同
            date_groups[date].append(i)

        
        return date_groups
    

class AdvancedCombineLoss(nn.Module):
    def __init__(self, alpha=0.00001*1e-7, lambda_const=100):
        super(AdvancedCombineLoss, self).__init__()
        self.alpha = alpha
        self.lambda_const = lambda_const
        self.mse_loss = nn.MSELoss()

    def forward(self, preds, targets):
        # 计算预测值与真实值之间的均方误差
        mse_loss = self.mse_loss(preds, targets)
        
        # 计算预测值和目标值的排名
        preds_rank = preds.argsort(descending=True).argsort()
        targets_rank = targets.argsort(descending=True).argsort()
        
        # 使用广播计算所有排名差异的组合
        rank_differences = preds_rank[:, None] - preds_rank[None, :]
        target_rank_differences = targets_rank[:, None] - targets_rank[None, :]
        
        # 计算预测排名一致性的惩罚项
        rank_loss = torch.maximum(
            torch.tensor(0.0, device=preds.device), 
            -rank_differences * target_rank_differences + self.lambda_const #容忍相差100名以内
        ) ** 2
        # 将两部分损失结合起来
        total_loss = mse_loss + self.alpha * (rank_loss.mean() - self.lambda_const ** 2)
        #print(f"mse_loss:{mse_loss},rank_loss:{self.alpha * (rank_loss.mean() - self.lambda_const ** 2)},total_loss:{total_loss}")
        return total_loss
    

class outlayer(nn.Module):
    def __init__(self, d_time, d_model, num_classes):
        super(outlayer, self).__init__()
        self.linear1 = nn.Linear(d_time, 1)
        self.linear2 = nn.Linear(2*d_model, num_classes)
        self.flat = nn.Flatten()
        
    def forward(self, x):
        y = x.permute(0,2,1)
        y = (self.linear1(y)).permute(0,2,1)
        y = torch.concat([y, x[:,-1,:].unsqueeze(-2)],dim = -2)
        y = self.linear2(self.flat(y))
        return y

class GRUModel(nn.Module):
    def __init__(self, config,sequence_len):
        super(GRUModel, self).__init__()
        self.hidden_size = config.hidden_size
        self.GRU_layer = nn.GRU(input_size=config.input_size, hidden_size=config.hidden_size, num_layers=config.lstm_layers, batch_first=True)
        self.GRU_layer1 = nn.GRU(input_size=config.hidden_size, hidden_size=config.hidden_size, num_layers=config.lstm_layers, batch_first=True)
        self.output_linear = outlayer(sequence_len, config.hidden_size, config.output_size)
        self.bn = nn.BatchNorm1d(config.hidden_size)
        self.hidden = None

    def forward(self, x):
        x, self.hidden = self.GRU_layer(x)
        x, self.hidden = self.GRU_layer1(x)
        x = self.bn(x.permute(0,2,1)).permute(0,2,1)
        x = self.output_linear(x)
        return x
    
class LSTMModel(nn.Module):
    def __init__(self, config):
        super(LSTMModel, self).__init__()
        self.lstm = LSTM(input_size=config.input_size, hidden_size=config.hidden_size,
                         num_layers=config.lstm_layers, batch_first=True)
        self.linear = Linear(in_features=config.hidden_size, out_features=config.output_size)

    def forward(self, x, hidden=None):
        lstm_out, hidden = self.lstm(x, hidden)
        linear_out = self.linear(lstm_out)
        return linear_out[:,-1,:], hidden #返回(batchsize, 1, 1)

def ic(Y_hat,Y):
    Y_hat =Y_hat.cpu()
    Y = Y.cpu()
    x = Y_hat.detach().numpy()
    y = Y.detach().numpy()

    # 计算相关系数矩阵
    corr_matrix = np.corrcoef(x.T, y.T)
    corr = corr_matrix[0, 1]
    return corr

# 定义计算斯皮尔曼相关系数的函数
def rankIc(Y_hat,Y):
    Y_hat =Y_hat.cpu()
    Y = Y.cpu()
    x = Y_hat.detach().numpy()
    y = Y.detach().numpy()
    return spearmanr(x, y)[0]

def draw_fig(config,train_loss,valid_loss,early_stop_epoch):
    # 创建一个图形实例
    plt.figure(figsize=(10, 6))

    # 绘制训练和验证损失
    plt.plot(train_loss, label='Train Loss')
    plt.plot(valid_loss, label='Validation Loss')

    plt.axvline(x=early_stop_epoch, color='r', linestyle='--', label='Early Stop')

    plt.legend()

    plt.title("Training and Validation Loss per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig(config.figure_save_path+f"loss.png")
    print("figure saved!")

def train(config, train_and_valid_data,train_and_valid_data_index):
    train_X,valid_X,train_Y,valid_Y = train_and_valid_data
    train_index,valid_index = train_and_valid_data_index

    train_custom_dataset = CustomDataset(train_X,train_Y,train_index)
    valid_custom_dataset = CustomDataset(valid_X,valid_Y,valid_index)

    # 使用 DataLoader 来加载数据集
    train_loader = DataLoader(train_custom_dataset, batch_size=config.batch_size,shuffle=False)
    valid_loader = DataLoader(valid_custom_dataset, batch_size=config.batch_size,shuffle=False)

    device = torch.device("cuda:2" if config.use_cuda and torch.cuda.is_available() else "cpu")

    model = LSTMModel(config).to(device)

    if config.add_train:                
        model.load_state_dict(torch.load(config.model_save_path + config.model_name))

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = AdvancedCombineLoss()     

    valid_loss_min = float("inf")
    valid_ic_min = float("-inf")
    bad_epoch = 0
    global_step = 0
    train_loss_total = []
    valid_loss_total = []
    early_stop_epoch = 0
    for epoch in range(config.epoch):
        model.train()                   
        train_loss_array = []
        train_ic_array = []
        train_rank_ic_array = []
        hidden_train = None
        for i, _data in enumerate(train_loader):
            _train_X, _train_Y = _data[0].to(device),_data[1].to(device)
            _train_X,_train_Y = _train_X.squeeze(dim=0),_train_Y.squeeze(dim=0)
            optimizer.zero_grad()               # 训练前要将梯度信息置 0
            if (_train_X.shape[0]==0):
                #print(i)
                continue
            #print(_train_X.shape)
            pred_Y, hidden_train = model(_train_X.float(), hidden_train)    # 这里走的就是前向计算forward函数

            hidden_train = None             # 如果非连续训练，把hidden重置即可

            loss = criterion(pred_Y.float(), _train_Y[:,-1,:].float())  # 计算loss
            loss.backward()                     # 将loss反向传播
            optimizer.step()                    # 用优化器更新参数
            train_loss_array.append(loss.item())
            cur_ic = ic(pred_Y,_train_Y[:,-1,:])
            rank_ic = rankIc(pred_Y,_train_Y[:,-1,:])
            train_ic_array.append(cur_ic)
            train_rank_ic_array.append(rank_ic)
            global_step += 1


        # 以下为早停机制，当模型训练连续config.patience个epoch都没有使验证集预测效果提升时，就停止，防止过拟合
        model.eval()                    # pytorch中，预测时要转换成预测模式
        valid_loss_array = []
        valid_ic_array = []
        valid_rank_ic_array = []
        hidden_valid = None
        for _valid_X, _valid_Y in valid_loader:
            _valid_X, _valid_Y = _valid_X.to(device), _valid_Y.to(device)
            _valid_X,_valid_Y = _valid_X.squeeze(dim=0),_valid_Y.squeeze(dim=0)
            if _valid_X.shape[0]==0:
                continue
            pred_Y, hidden_valid = model(_valid_X.float(), hidden_valid)
            hidden_valid = None
            loss = criterion(pred_Y.float(), _valid_Y[:,-1,:].float())  # 验证过程只有前向计算，无反向传播过程
            valid_loss_array.append(loss.item())
            cur_ic = ic(pred_Y,_valid_Y[:,-1,:])
            rank_ic = rankIc(pred_Y,_valid_Y[:,-1,:])
            valid_ic_array.append(cur_ic)
            valid_rank_ic_array.append(rank_ic)

        train_loss_cur = np.mean(train_loss_array)
        valid_loss_cur = np.mean(valid_loss_array)
        train_loss_total.append(train_loss_cur)
        valid_loss_total.append(valid_loss_cur)

        train_ic_cur = np.mean(train_ic_array)
        valid_ic_cur = np.mean(valid_ic_array)
        train_rank_ic_cur = np.mean(train_rank_ic_array)
        valid_rank_ic_cur = np.mean(valid_rank_ic_array)
        print(f"epoch:{epoch}")
        print(f"the train loss is {train_loss_cur}.the train ic is {train_ic_cur,train_rank_ic_cur}")
        print(f"the valid loss is {valid_loss_cur}.the valid ic is {valid_ic_cur,valid_rank_ic_cur}")

        if train_rank_ic_cur>0.02 and valid_rank_ic_cur>0.02:
            if valid_rank_ic_cur > valid_ic_min:
                valid_ic_min = valid_rank_ic_cur
                torch.save(model.state_dict(), config.model_save_path + config.model_name)
                print("model saved!")

        if valid_loss_cur < valid_loss_min:
            valid_loss_min = valid_loss_cur
            bad_epoch = 0
            early_stop_epoch = epoch
            torch.save(model.state_dict(), config.model_save_path + config.model_name)  # 模型保存
        else:
            bad_epoch += 1
            print(bad_epoch)
            if bad_epoch >= config.patience:    # 如果验证集指标连续patience个epoch没有提升，就停掉训练
                print(" The training stops early in epoch {}".format(epoch))
                break

    draw_fig(config,train_loss_total,valid_loss_total,early_stop_epoch)

def predict(config, test_X, test_Y, test_index):
    # 获取测试数据
    test_custom_dataset = CustomDataset(test_X, Y=None, index=test_index)
    # 使用 DataLoader 来加载数据集
    test_loader = DataLoader(test_custom_dataset, batch_size=config.batch_size,shuffle=False)
    # 加载模型
    device = torch.device("cuda:2" if config.use_cuda and torch.cuda.is_available() else "cpu")
    
    model = LSTMModel(config).to(device)
    
    model.load_state_dict(torch.load(config.model_save_path + config.model_name))   # 加载模型参数

    # 预测过程
    model.eval()
    hidden_predict = None
    result = dict()
    for _test_X,date in test_loader:
        _test_X = _test_X.to(device)
        _test_X = _test_X.squeeze(dim=0)

        if _test_X.shape[0]==0:
                continue
        with torch.no_grad():
            #print(_test_X.shape)
            pred_X, hidden_predict = model(_test_X.float(), hidden_predict)
            hidden_predict = None
        result[date[0]] = pred_X.detach().cpu().numpy()

    return result # 先去梯度信息，如果在gpu要转到cpu，最后要返回numpy数据