import torch
from torch.nn import Linear
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import torch.nn as nn

from scipy.stats import spearmanr
from torch.utils.data import Dataset

import matplotlib.pyplot as plt
torch.manual_seed(42)

class CustomDataset(Dataset):
    def __init__(self, X,Y=None):
        # 初始化函数，在这里设置数据集的初始参数
        self.X = X
        self.Y = Y
        self.len = len(self.X.index.get_level_values(level='date').unique())
        self.date_list = self.X.index.get_level_values(level='date').unique().to_list()

    def __len__(self):
        # 返回数据集中的样本数
        return self.len

    def __getitem__(self, index):
        # 生成单个样本
        # 单个样本即为截面数据
        date = self.date_list[index]
        sample_X = self.X.loc[date].values
        if not isinstance(self.Y,type(None)):
            sample_Y = self.Y.loc[date].values.astype(float)

            return (sample_X,sample_Y)
        else:
            return sample_X


class MLP(nn.Module):
    def __init__(self, config):
        super(MLP, self).__init__()
        
        self.layers = nn.ModuleList()  # 创建一个模块列表以存储所有层
        self.activations = nn.ModuleList()
        self.batch_norms = nn.ModuleList() if config.use_batch_norm else None
        self.dropouts = nn.ModuleList() if config.use_dropout else None
        # 构建网络层
        prev_size = config.input_size
        if config.init_fns!=[]:
            i = 0
            for size, activation_fn, init_fn in zip(config.layer_sizes, config.activation_fns, config.init_fns):
                layer = nn.Linear(prev_size, size)
                init_fn(layer.weight)  # 应用初始化函数
                self.layers.append(layer)
                if config.use_batch_norm:
                    self.batch_norms.append(nn.BatchNorm1d(size))

                if config.use_dropout:
                    self.dropouts.append(nn.Dropout(config.dropout_rates[i]))
                self.activations.append(activation_fn())
                prev_size = size
                i+=1
        else:
            i = 0
            for size,activation_fn in zip(config.layer_sizes,config.activation_fns):
                self.layers.append(nn.Linear(prev_size, size))
                if config.use_batch_norm:
                    self.batch_norms.append(nn.BatchNorm1d(size))

                if config.use_dropout:
                    self.dropouts.append(nn.Dropout(config.dropout_rates[i]))
                self.activations.append(activation_fn())
                prev_size = size
                i = i+1

    def forward(self, x):
        for i,layer in enumerate(self.layers):
            x = layer(x)
            if self.batch_norms:
                x=self.batch_norms[i](x)
            x = self.activations[i](x)
            if self.dropouts:
                x=self.dropouts[i](x)
        return x
    
class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.00001*1e-7, lambda_const=100):
        super(CombinedLoss, self).__init__()
        self.alpha = alpha
        self.lambda_const = lambda_const
        self.mse_loss = nn.MSELoss()

    def forward(self, preds, targets):
        # 计算预测值与真实值之间的均方误差
        mse_loss = self.mse_loss(preds, targets)
        
        # 使用广播计算所有排名差异的组合
        rank_differences = preds[:, None] - preds[None, :]
        target_rank_differences = targets[:, None] - targets[None, :]
        
        # 计算预测排名一致性的惩罚项
        rank_loss = torch.maximum(
            torch.tensor(0.0, device=preds.device), 
            -rank_differences * target_rank_differences + self.lambda_const #容忍相差100名以内
        ) ** 2
        # 将两部分损失结合起来
        total_loss = mse_loss + self.alpha * (rank_loss.mean() - self.lambda_const ** 2)
        #total_loss = self.alpha * (rank_loss.mean() - self.lambda_const ** 2)
        #print(f"mse_loss:{mse_loss},rank_loss:{self.alpha * (rank_loss.mean() - self.lambda_const ** 2)},total_loss:{total_loss}")
        return total_loss


class ICLoss(nn.Module):
    def __init__(self,diversity_factor=0.00001):
        super(ICLoss, self).__init__()
        self.diversity_factor = diversity_factor

    def forward(self, preds, targets):

        # 计算皮尔逊相关系数作为IC的近似
        preds_mean = preds.mean()
        targets_mean = targets.mean()
        covariance = ((preds - preds_mean) * (targets - targets_mean)).mean()
        preds_std = preds.std()
        targets_std = targets.std()

        corr = covariance / (preds_std * targets_std)

        # 添加多样性正则化项
        # 假设 preds 是一个批量预测值，我们想鼓励这个批量中不同的预测值
        unique_preds = torch.unique(preds)
        diversity_loss = -self.diversity_factor * unique_preds.numel()

        print(f"corr:{corr},unique_preds:{unique_preds},diversity_loss:{diversity_loss}")

        total_loss = corr + diversity_loss
        return -total_loss
    
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
    
#自定义对数均方误差损失函数
class LogCoshLoss(nn.Module):
    def __init__(self):
        super(LogCoshLoss, self).__init__()

    def forward(self, input, target):
        epsilon = 1e-6
        #print(f"input:{input},target:{target}")
        input = (input+abs(input.min()))
        log_pred = torch.log(input + 1 + epsilon)
        #print(f"log_pred:{log_pred}")
        target = (target+abs(target.min()))
        log_true = torch.log(target + 1 + epsilon)
        loss = torch.sqrt(torch.mean(torch.pow(log_pred - log_true, 2)))
        return loss
    
class WeightedClassLoss_decay(nn.Module):   
    def __init__(self,num_classes=5,highest_weight=1.0,decay_factor=0.5):
        super(WeightedClassLoss_decay, self).__init__()
        self.num_classes = num_classes
        # 设置最高收益率等级的权重为1
        self.class_weights = {num_classes: highest_weight}
        
        # 为其他收益率等级设置半衰加权权重
        for i in range(num_classes - 1, 0, -1):
            self.class_weights[i] = self.class_weights[i + 1] * decay_factor 
        self.device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

    def forward(self, predictions, targets):
        # 获取每个targets对应的权重
        quantile_targets = self.get_quantiles(targets, self.num_classes)
        weights = torch.tensor([self.class_weights[q.item()] for q in quantile_targets],device=self.device)
        # 计算加权的均方误差
        loss = torch.mean(weights * (predictions - targets) ** 2)
        return loss
    
    def get_quantiles(self,tensor, num_quantiles):
        # 计算分位数值
        quantiles = torch.quantile(tensor, torch.linspace(0, 1, steps=num_quantiles+1,device=self.device))
        # 生成分位数标签，从1到num_quantiles
        quantile_labels = torch.searchsorted(quantiles, tensor, right=False)
        #如果quantile_labels为0，则将其设置为1
        quantile_labels[quantile_labels == 0] = 1
        return quantile_labels
      
class WeightedClassLoss(nn.Module):
    def __init__(self, num_classes=5, highest_weight=1.0, decay_factor=0.5):
        super(WeightedClassLoss, self).__init__()
        self.num_classes = num_classes
        self.device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
        
        # 两头高，中间低的权重分配
        # 创建一个倒三角形分布权重，两边高，中间低
        weights = np.linspace(start=highest_weight, stop=0, num=(num_classes+1)//2, endpoint=False)
        if num_classes % 2 == 0:  # 如果类别数是偶数，确保中间两个类别的权重相同
            weights = np.concatenate((weights, weights[::-1]))
        else:  # 如果类别数是奇数，中间类别权重为0
            weights = np.concatenate((weights, [0], weights[::-1]))
        
        # 将权重转换为字典形式并存储
        self.class_weights = {i+1: weight for i, weight in enumerate(weights)}
        
    
    def forward(self, predictions, targets):
        # 获取每个targets对应的权重
        quantile_targets = self.get_quantiles(targets, self.num_classes)
        weights = torch.tensor([self.class_weights[q.item()] for q in quantile_targets],device=self.device)
        # 计算加权的均方误差
        loss = torch.mean(weights * (predictions - targets) ** 2)
        return loss
    
    def get_quantiles(self,tensor, num_quantiles):
        # 计算分位数值
        quantiles = torch.quantile(tensor, torch.linspace(0, 1, steps=num_quantiles+1,device=self.device))
        # 生成分位数标签，从1到num_quantiles
        quantile_labels = torch.searchsorted(quantiles, tensor, right=False)
        #如果quantile_labels为0，则将其设置为1
        quantile_labels[quantile_labels == 0] = 1
        return quantile_labels

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
    plt.savefig(config.figure_save_path+f"{config.frame}")
    print("figure saved!")

def train(config, train_and_valid_data):
    train_X,valid_X,train_Y,valid_Y = train_and_valid_data

    train_custom_dataset = CustomDataset(train_X,train_Y)
    valid_custom_dataset = CustomDataset(valid_X,valid_Y)

    # 使用 DataLoader 来加载数据集
    train_loader = DataLoader(train_custom_dataset, batch_size=config.batch_size,shuffle=False)
    valid_loader = DataLoader(valid_custom_dataset, batch_size=config.batch_size,shuffle=False)

    device = torch.device("cuda:2" if config.use_cuda and torch.cuda.is_available() else "cpu")

    model = MLP(config).to(device)
    print(model)

    if config.add_train:                
        model.load_state_dict(torch.load(config.model_save_path + config.model_name))

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = WeightedClassLoss_decay()

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
        for i, _data in enumerate(train_loader):
            _train_X, _train_Y = _data[0].to(device),_data[1].to(device)
            _train_X,_train_Y = _train_X.squeeze(dim=0),_train_Y.squeeze(dim=0)
            optimizer.zero_grad()               # 训练前要将梯度信息置 0
            if (_train_X.shape[0]==0):
                #print(i)
                continue
            pred_Y = model(_train_X.float())    # 这里走的就是前向计算forward函数
            pred_Y = pred_Y.squeeze()
            _train_Y = _train_Y.squeeze()
            loss = criterion(pred_Y.float(), _train_Y.float())  # 计算loss
            loss.backward()                     # 将loss反向传播
            optimizer.step()                    # 用优化器更新参数
            train_loss_array.append(loss.item())
            cur_ic = ic(pred_Y,_train_Y)
            rank_ic = rankIc(pred_Y,_train_Y)
            train_ic_array.append(cur_ic)
            train_rank_ic_array.append(rank_ic)
            global_step += 1

        # 以下为早停机制，当模型训练连续config.patience个epoch都没有使验证集预测效果提升时，就停止，防止过拟合
        model.eval()                    # pytorch中，预测时要转换成预测模式
        valid_loss_array = []
        valid_ic_array = []
        valid_rank_ic_array = []
        for _valid_X, _valid_Y in valid_loader:
            _valid_X, _valid_Y = _valid_X.to(device), _valid_Y.to(device)
            _valid_X,_valid_Y = _valid_X.squeeze(dim=0),_valid_Y.squeeze(dim=0)
            if _valid_X.shape[0]==0:
                continue
            pred_Y= model(_valid_X.float())
            _valid_Y = _valid_Y.squeeze()
            pred_Y = pred_Y.squeeze()
            _valid_Y = _valid_Y.squeeze()
            loss = criterion(pred_Y.float(), _valid_Y.float())  # 验证过程只有前向计算，无反向传播过程
            valid_loss_array.append(loss.item())
            cur_ic = ic(pred_Y,_valid_Y)
            rank_ic = rankIc(pred_Y,_valid_Y)
            valid_ic_array.append(cur_ic)
            valid_rank_ic_array.append(rank_ic)
        
        train_loss_cur = np.mean(train_loss_array)
        valid_loss_cur = np.mean(valid_loss_array)

        train_loss_total.append(train_loss_cur)
        valid_loss_total.append(valid_loss_cur)

        train_ic_cur = np.mean(train_ic_array)
        train_rank_ic_cur = np.mean(train_rank_ic_array)
        valid_ic_cur = np.mean(valid_ic_array)
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
            torch.save(model.state_dict(), config.model_save_path + config.model_name)
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
    test_custom_dataset = CustomDataset(test_X)

    # 使用 DataLoader 来加载数据集
    test_loader = DataLoader(test_custom_dataset, batch_size=config.batch_size,shuffle=False)

    device = torch.device("cuda:2" if config.use_cuda and torch.cuda.is_available() else "cpu")

    model = MLP(config).to(device)

    model.load_state_dict(torch.load(config.model_save_path + config.model_name))   # 加载模型参数

    # 先定义一个tensor保存预测结果
    result = torch.Tensor().to(device)

    # 预测过程
    model.eval()
    for _data in test_loader:
        data_X = _data[0].to(device)
        with torch.no_grad():
            pred_X = model(data_X.float())
        cur_pred = torch.squeeze(pred_X, dim=0)
        result = torch.cat((result, cur_pred), dim=0)

    return result.detach().cpu().numpy()    # 先去梯度信息，如果在gpu要转到cpu，最后要返回numpy数据