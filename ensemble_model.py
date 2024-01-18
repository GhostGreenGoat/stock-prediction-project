#%%
import torch
from torch.nn import Linear
from torch.utils.data import DataLoader, TensorDataset

import numpy as np
import torch.nn as nn

from scipy.stats import spearmanr
from torch.utils.data import Dataset

import matplotlib.pyplot as plt

import torch.nn.functional as F
import os

import pandas as pd
#from main import Config
from model.model_mlp import MLP,predict
torch.manual_seed(42)

class Config:
    random_seed = 42 
    #规定训练的参数
    valid_data_rate = 0.15

    #规定选用的特征集合。all,am,pm分别表示全天，早盘，晚盘
    selected_feature = "all"

    debug_mode = False #True为debug模式，采用随机抽取的100支股票集合
    
    #训练集结束的日期和测试集开始的日期
    train_end_date = '20211231'
    test_start_date = '20220104'
    
    #规定训练标签,股票池
    label = 'y1_label'
    universe = 'univ_tradable'

    #规定预测未来几天
    predict_day = 1

    #模型参数
    input_size = 249
    layer_sizes = [128,1]
    activation_fns = [nn.Tanh,nn.Tanh]
    use_batch_norm = True
    use_dropout = True
    dropout_rates = [0.2,0.2]
    activate_fn = 'tanh'
    init_fns = [] #配合激活函数使用相应的参数初始化方法，共有：nn.init.xavier_normal_，nn.init.kaiming_normal_
    #若没有指定，pytorch默认使用Lecun Initialization
    label_normalization = 'rank' #取值有profile,rank和空值
    feature_normal  = 'normal'

    #训练参数
    learning_rate = 0.001
    epoch = 250 #不考虑早停的前提下整个模型训练多少遍
    patience = 10
    batch_size = 1000

    #训练方式（是否增量训练）
    add_train = False
    do_train = False
    do_validation = False
    do_predict = True
    shuffle_train_data = False
    use_cuda = True

    #模型框架
    str_list = [str(i) for i in layer_sizes]
    size="_".join(str_list)
    loss = "advance"
    subframe = f"{label}_{label_normalization}_{selected_feature}_{feature_normal}"
    if not use_dropout and not use_batch_norm:
        frame = f"selected_feature({selected_feature})_loss({loss})_depth{len(layer_sizes)}_sizes({size})"
    elif use_batch_norm and not use_dropout:
        frame = f"selected_feature({selected_feature})_loss({loss})_depth{len(layer_sizes)}_sizes({size})_batchnorm"
    elif use_dropout and not use_batch_norm:
        frame = f"selected_feature({selected_feature})_loss({loss})_depth{len(layer_sizes)}_sizes({size})_dropout"
    else:
        frame = f"selected_feature({selected_feature})_loss({loss})_depth{len(layer_sizes)}_sizes({size})_bd"
    #路径参数
    train_data_path = "/home/laiminzhi/wenbin/DL_stock_combo/data/xy_data/xy_data.h5"
    model_save_path = f"/home/laiminzhi/reconfiguration_code/ensemble/"
    figure_save_path = f"/home/laiminzhi/reconfiguration_code/figure/{subframe}/"
    predict_save_path = f"/home/laiminzhi/reconfiguration_code/ensemble/result/train/"
    valid_save_path = f"/home/laiminzhi/reconfiguration_code/predict_data/{subframe}/{frame}_valid/"
    processed_train_data_path = f"/home/laiminzhi/reconfiguration_code/train_data/"+ subframe +"/"
    processed_test_data_path = f"/home/laiminzhi/reconfiguration_code/test_data/"+ subframe+"/"
    
    save_processed_train_data = False #如果已经有处理好的train_data,这项为false，若为True则会从头处理train data，保存到processed_train_data_path中

    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)    # makedirs 递归创建目录
    if not os.path.exists(figure_save_path):
        os.makedirs(figure_save_path)
    if not os.path.exists(predict_save_path):
        os.makedirs(predict_save_path)
    if not os.path.exists(processed_train_data_path):
        os.makedirs(processed_train_data_path)
    if not os.path.exists(processed_test_data_path):
        os.makedirs(processed_test_data_path)
    if not os.path.exists(valid_save_path):
        os.makedirs(valid_save_path)
    #名称
    model_name = "ensemblemodel"+ ".pth"

def load_train_data(path):
    train_x = pd.read_hdf(os.path.join(path,'train_x.h5'))
    valid_x = pd.read_hdf(os.path.join(path,'valid_x.h5'))
    train_y = pd.read_hdf(os.path.join(path,"train_y.h5"))
    valid_y = pd.read_hdf(os.path.join(path,"valid_y.h5"))

    train_and_valid_data = (train_x,valid_x,train_y,valid_y)

    return train_and_valid_data
#%%
class EnsembleModel(nn.Module):
    def __init__(self, num_models):
        super(EnsembleModel, self).__init__()
        # 定义一个线性层，权重为各模型的系数
        self.coefficients = nn.Linear(num_models, 1, bias=False)
        
        # 初始化权重为正值
        self.coefficients.weight.data.fill_(1.0/num_models)

    def forward(self, model_outputs):
        # 应用线性层
        #print(f"model_outputs:{model_outputs}")
        final_output = self.coefficients(model_outputs)
        return final_output
    
def get_subdirectories(path):
    """ 返回指定路径下所有子文件夹的路径列表 """
    subdirectories = [os.path.join(path, name) for name in os.listdir(path)
                      if os.path.isdir(os.path.join(path, name))]
    return subdirectories

def get_model_outputs(path_list):
    outputs = []
    for path in path_list:
        model_names = os.listdir(path)
        for model_name in model_names:
            output = np.load(os.path.join(path,model_name))
            outputs.append(output)
    return outputs

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

#%%
#读取已有的模型进行集成模型的训练
config = Config()
#定义路径参数
train_data_path = config.processed_train_data_path
model_result_path = '/home/laiminzhi/reconfiguration_code/train_result/'
subdir = get_subdirectories(model_result_path)

#%%
# 加载训练数据
model_outputs = get_model_outputs(subdir)
num_models = len(model_outputs)

model_outputs = np.array(model_outputs)#(6,2399342,1)
X = model_outputs.transpose(1,0,2)
train_x = np.squeeze(X,axis=-1)

#加载valid数据
valid_result_path = '/home/laiminzhi/reconfiguration_code/valid_result/'
subdir = get_subdirectories(valid_result_path)
valid_outputs = get_model_outputs(subdir)
valid_outputs = np.array(valid_outputs)
valid_x = valid_outputs.transpose(1,0,2)
valid_x = np.squeeze(valid_x,axis=-1)


#加载label数据
train_and_valid_data = load_train_data(config.processed_train_data_path)
train_y = np.array(train_and_valid_data[2].values)
valid_y = np.array(train_and_valid_data[3].values)

valid_y_df = train_and_valid_data[3]
train_y_df = train_and_valid_data[2]
#%%
def train_ensemble_model(config,train_x,train_y,valid_x,valid_y,num_models):
    device = torch.device("cuda:2")

    train_X, train_Y = torch.from_numpy(train_x).float(), torch.from_numpy(train_y).float()
    train_loader = DataLoader(TensorDataset(train_X, train_Y), batch_size=config.batch_size)
    valid_X, valid_Y = torch.from_numpy(valid_x).float(), torch.from_numpy(valid_y).float()
    valid_loader = DataLoader(TensorDataset(valid_X, valid_Y), batch_size=config.batch_size)
    model = EnsembleModel(num_models).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.MSELoss()

    valid_loss_min = float("inf")
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
            #_train_X,_train_Y = _train_X.squeeze(dim=0),_train_Y.squeeze(dim=0)
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
            #限制系数大于等于0
            with torch.no_grad():
                model.coefficients.weight.data.clamp_(min=0)
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
            #_valid_X,_valid_Y = _valid_X.squeeze(dim=0),_valid_Y.squeeze(dim=0)
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


# %%
def predict_ensemble_model(config, test_X, num_models):
    # 获取测试数据
    test_X = torch.from_numpy(test_X).float()
    # 使用 DataLoader 来加载数据集
    test_loader = DataLoader(TensorDataset(test_X), batch_size=config.batch_size,shuffle=False)

    device = torch.device("cuda:2")

    model = EnsembleModel(num_models).to(device)

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

def calculate_spearman(group):
    return spearmanr(group['Y_hat'], group['Y'])[0]


def save_test_data(Y_hat,test_Y,output_dir):
    """
    input:
        Y_hat为predict函数的输出结果，即模型预测结果，形状为(n,),n为样本数量
        test_Y为实际的Y,为seires
    output:
        以predict_date为文件名将每一天的Y_hat和test_Y存储到output_dir中，并且求出每天截面的ic，再在时序上求均值后输出
    """
    n = test_Y.shape[0]
    if isinstance(test_Y,pd.Series):
        df = test_Y.to_frame(name='Y')
    else:
        df = test_Y
        df.columns=['Y']
    df['Y_hat'] = Y_hat
    grouped_df = df.groupby(level='date')
    
    for name,group in grouped_df:
        group = group.reset_index(drop=False)
        group.to_csv(output_dir+name+'.csv')
    print("data saved!")
    
    #计算截面ic
    daily_correlation = grouped_df.apply(lambda x:x['Y_hat'].corr(x['Y']))
    average_coor = daily_correlation.mean()
    #计算rank ic
    spearman_correlations = df.groupby(level='date').apply(calculate_spearman)
    rank_avg = spearman_correlations.mean()
    return (average_coor,rank_avg)

if config.do_train:
    train_ensemble_model(config,train_x,train_y,valid_x,valid_y,num_models)

model = EnsembleModel(num_models)
model.load_state_dict(torch.load(config.model_save_path + config.model_name))
for name, param in model.named_parameters():
            if param.requires_grad:
                print(name, param.data)
print(model)

if config.do_predict:
    Y_hat = predict_ensemble_model(config,train_x,num_models)
    ic = save_test_data(Y_hat,train_y_df,config.predict_save_path)
    print(f"ic={ic[0]},rank ic = {ic[1]}")