import torch
from torch.nn import Module, LSTM, Linear
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import torch.nn as nn

from torch.utils.data import Dataset
torch.manual_seed(42)
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split

class CustomDataset(Dataset):
    def __init__(self,timestep,X,Y=None):
        # 初始化函数，在这里设置数据集的初始参数
        self.timestep=timestep
        self.X = X
        self.Y = Y
        self.len = self.X.shape[1]

    def __len__(self):
        # 返回数据集中的样本数
        return self.len

    def __getitem__(self,di):
        sample_x = self.get_slices(self.X,self.timestep,di)
        if not isinstance(self.Y,type(None)):
            sample_y = self.get_slices(self.Y,self.timestep,di)
            return sample_x,sample_y
        else:
            return sample_x
    
    def get_slices(self, data, timestep, di):
        #不需要universe，直接从已经对齐的(featrue,date,stock)数据中获取timestep的切片
        #输出:(2836,243,30)
        #要求输入的di是大于timestep-1的，且di在data.shape[1]范围内,di对应每个timestep的最后一天
        res = [None] * data.shape[2]
        if di < timestep-1:
            return np.zeros((data.shape[2],timestep,data.shape[0]))
        else:
            for ii in range(data.shape[2]):
                # 在前面universe不为0的日子中搜索该股票，找不到就保持-1
                # 使用np.where查找universe中的非零值
                
                values = data[:,(di+1)-timestep:(di+1),ii]#(243,timestep,1)
                res[ii] = values  
        res = np.array(res)
        res = res.transpose(0,2,1)             
        return np.array(res) #(2836,30,243)
    

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

class no_udlimit_wmse(nn.Module):
    def __init__(self,alpha=0.01):
        super(no_udlimit_wmse, self).__init__()
        self.alpha = alpha

    def forward(self, predict, target, weight):
        """
        计算带权重的均方误差（WMSE）
        
        :param predict: 预测值，形状为 (code, 1)
        :param target: 真实值，形状为 (code, 1, 1)
        :param weight: 权重条件，形状为 (code, 1)
        :return: 加权均方误差
        """
        # 确保predict和target的形状一致
        target = target.squeeze()
        
        # 设置权重：如果weight的值为1，则权重设置为0.01，否则设置为1
        weights = torch.where(weight == 1, torch.tensor(self.alpha,device=weight.device), torch.tensor(1.0,device=weight.device))
        
        # 计算误差（差的平方）
        errors = (predict - target) ** 2
        
        # 应用权重
        weighted_errors = errors * weights
        
        # 计算WMSE
        wmse = weighted_errors.mean()
        
        return wmse

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
    def __init__(self, config):
        super(GRUModel, self).__init__()
        self.hidden_size = config.hidden_size
        self.GRU_layer = nn.GRU(input_size=config.input_size, hidden_size=config.hidden_size, num_layers=config.lstm_layers, batch_first=True)
        self.GRU_layer1 = nn.GRU(input_size=config.hidden_size, hidden_size=config.hidden_size, num_layers=config.lstm_layers, batch_first=True)
        self.output_linear = outlayer(config.windowsize, config.hidden_size, config.output_size)
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
                         num_layers=config.lstm_layers, batch_first=True, dropout=config.dropout_rate)
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

def train(config, train_and_valid_data):
    train_X,valid_X,train_Y,valid_Y = train_and_valid_data

    train_custom_dataset = CustomDataset(config.windowsize,train_X,train_Y)
    valid_custom_dataset = CustomDataset(config.windowsize,valid_X,valid_Y)

    # 使用 DataLoader 来加载数据集
    train_loader = DataLoader(train_custom_dataset, batch_size=config.batch_size,shuffle=False)
    valid_loader = DataLoader(valid_custom_dataset, batch_size=config.batch_size,shuffle=False)

    device = torch.device("cuda:2" if config.use_cuda and torch.cuda.is_available() else "cpu")
    print("device:",device)

    model = GRUModel(config).to(device)
    print(model)
    #print(next(model.parameters()).is_cuda)
    if config.add_train:                
        model.load_state_dict(torch.load(config.model_save_path + config.model_name))
        print("add train...")

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = no_udlimit_wmse()

    #给涨停股票降低权重
    #读入涨停股票的数据
    ud_limit = pd.read_hdf(f"/home/laiminzhi/data/xydata/{config.data}_udlimit.h5")
    ud_limit = pd.pivot_table(ud_limit,index = "date",columns = "code",values = "ud_limit_h2")
    unique_dates = ud_limit.index.unique()

    train_dates,valid_dates = train_test_split(unique_dates,test_size = config.valid_data_rate,
                                                           random_state= config.random_seed, shuffle= config.shuffle_train_data)
    ud_limit_train = ud_limit.loc[ud_limit.index.isin(train_dates)]
    ud_limit_valid = ud_limit.loc[ud_limit.index.isin(valid_dates)]
    #转换为numpy数组
    ud_limit_train = np.array(ud_limit.values)
    ud_limit_valid = np.array(ud_limit.values)
    
    valid_loss_min = float("inf")
    valid_ic_min = float("-inf")
    bad_epoch = 0
    global_step = 0
    train_loss_total = []
    valid_loss_total = []
    early_stop_epoch = 0
    for epoch in range(config.epoch):
        print(f"epoch:{epoch}")
        model.train()                   
        train_loss_array = []
        train_ic_array = []
        train_rank_ic_array = []
        hidden_train = None
        print("training...")
        for i, _data in tqdm(enumerate(train_loader)):
            if i<config.windowsize-1:
                continue
            _train_X, _train_Y = _data[0].to(device),_data[1].to(device)
            _train_X,_train_Y = _train_X.squeeze(dim=0),_train_Y.squeeze(dim=0)
            optimizer.zero_grad()               # 训练前要将梯度信息置 0
            if (_train_X.shape[0]==0):
                #print(i)
                continue
            #print(_train_X.shape)
            #print(_train_X.is_cuda,_train_Y.is_cuda)
            #pred_Y, hidden_train = model(_train_X.float(), hidden_train)    # 这里走的就是前向计算forward函数
            pred_Y = model(_train_X.float())

            hidden_train = None             # 如果非连续训练，把hidden重置即可

            #weight为ud_limit当天的数据
            weight = torch.tensor(ud_limit_train[i]).to(device)

            loss = criterion(pred_Y.float(), _train_Y[:,-1,:].float(),weight)  # 计算loss
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
        print("validating...")
        for i,(_valid_X, _valid_Y) in tqdm(enumerate(valid_loader)):
            if i<config.windowsize-1:
                continue
            _valid_X, _valid_Y = _valid_X.to(device), _valid_Y.to(device)
            _valid_X,_valid_Y = _valid_X.squeeze(dim=0),_valid_Y.squeeze(dim=0)
            if _valid_X.shape[0]==0:
                continue
            #pred_Y, hidden_valid = model(_valid_X.float(), hidden_valid)
            pred_Y = model(_valid_X.float())
            hidden_valid = None
            
            weight = torch.tensor(ud_limit_valid[i]).to(device)

            loss = criterion(pred_Y.float(), _valid_Y[:,-1,:].float(),weight)  # 验证过程只有前向计算，无反向传播过程
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
            print("model saved!")
        else:
            bad_epoch += 1
            print(bad_epoch)
            if bad_epoch >= config.patience:    # 如果验证集指标连续patience个epoch没有提升，就停掉训练
                print(" The training stops early in epoch {}".format(epoch))
                break

    draw_fig(config,train_loss_total,valid_loss_total,early_stop_epoch)

def predict(config, test_X):
    # 获取测试数据
    test_custom_dataset = CustomDataset(config.windowsize,test_X, Y=None)
    # 使用 DataLoader 来加载数据集
    test_loader = DataLoader(test_custom_dataset, batch_size=config.batch_size,shuffle=False)
    # 加载模型
    device = torch.device("cuda:2" if config.use_cuda and torch.cuda.is_available() else "cpu")
    
    model = LSTMModel(config).to(device)
    
    model.load_state_dict(torch.load(config.model_save_path + config.model_name))   # 加载模型参数

    # 预测过程
    model.eval()
    hidden_predict = None
    result = []
    for _test_X in test_loader:
        _test_X = _test_X.to(device)
        _test_X = _test_X.squeeze(dim=0)

        if _test_X.shape[0]==0:
                continue
        with torch.no_grad():
            #print(_test_X.shape)
            #pred_X, hidden_predict = model(_test_X.float(), hidden_predict)
            pred_X = model(_test_X.float())
            hidden_predict = None
        
        result.append(pred_X.cpu().numpy())

    return result # 先去梯度信息，如果在gpu要转到cpu，最后要返回numpy数据