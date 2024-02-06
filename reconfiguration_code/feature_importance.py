#%%
import pandas as pd
import numpy as np
import os
import sys
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import QuantileTransformer
import torch.nn as nn
import torch
from scipy.stats import spearmanr
from captum.attr import IntegratedGradients
np.random.seed(42)
torch.manual_seed(42)
from model.model_mlp import MLP


class Config:
    random_seed = 42 
    #规定选用的特征集合。all,am,pm分别表示全天，早盘，晚盘
    selected_feature = "pm"
    
    #训练集结束的日期和测试集开始的日期
    train_end_date = '20211231'
    test_start_date = '20220104'
    
    #规定训练标签,股票池
    label = 'y1_label'
    universe = 'univ_tradable'
    limit = 'ud_limit_h2' #去掉的涨跌停数据

    #规定预测未来几天
    predict_day = 1

    #模型参数
    input_size = 249
    layer_sizes = [249,128,128,1]
    activation_fns = [nn.Tanh,nn.Tanh,nn.Tanh,nn.Tanh]
    use_batch_norm = True
    use_dropout = True
    dropout_rates = [0.2,0.2,0.2,0.2]
    activate_fn = 'tanh'
    init_fns = [] #配合激活函数使用相应的参数初始化方法，共有：nn.init.xavier_normal_，nn.init.kaiming_normal_
    #若没有指定，pytorch默认使用Lecun Initialization
    label_normalization = '' #取值有profile,rank和空值
    feature_nomalization = 'minmax'

    #模型框架
    str_list = [str(i) for i in layer_sizes]
    size="_".join(str_list)
    loss = "advance"
    subframe = f"{label}_{label_normalization}_{selected_feature}_{feature_nomalization}_no_{limit}"
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
    model_save_path = f"/home/laiminzhi/reconfiguration_code/model/{subframe}/"
    figure_save_path = f"/home/laiminzhi/reconfiguration_code/figure/{subframe}/"
    predict_save_path = f"/home/laiminzhi/reconfiguration_code/predict_data/{subframe}/{frame}/"
    valid_save_path = f"/home/laiminzhi/reconfiguration_code/predict_data/{subframe}/{frame}_valid/"
    processed_train_data_path = f"/home/laiminzhi/reconfiguration_code/train_data/"+ subframe +"/"
    processed_test_data_path = f"/home/laiminzhi/reconfiguration_code/test_data/"+ subframe+"/"
    

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
    model_name = "model_"+frame + ".pth"

#%%
config = Config()

torch.manual_seed(42)
np.random.seed(42)

test_X = pd.read_hdf(config.processed_train_data_path+'valid_x.h5')
input = test_X.iloc[0,:].values
Y_hat = pd.read_hdf(config.processed_train_data_path+'valid_y.h5')
config.input_size = input.shape[0]
model = MLP(config)
model.eval()
ig = IntegratedGradients(model)
input = torch.from_numpy(input).float()
input = input.unsqueeze(0)

def do_attributions(input_df,baseline):
    #input为一个样本，是df
    for i in range(input_df.shape[0]):
        input = input_df.iloc[i,:].values
        input = torch.from_numpy(input).float()
        input = input.unsqueeze(0)
        attributions,delta = ig.attribute(input,baselines=baseline,return_convergence_delta=True)
        if i == 0:
            attributions_df = pd.DataFrame(attributions.detach().numpy())
        else:
            attributions = attributions.detach().numpy()
            attributions_df = pd.concat([attributions_df,pd.DataFrame(attributions)],axis=0,ignore_index=True)
    return attributions_df
#%%
tmp = test_X.sample(n=10000,random_state=42)
#%%
baseline = torch.zeros(input.shape)
attributions_df = do_attributions(tmp,baseline)
attributions_df.columns = test_X.columns
mean = attributions_df.mean(axis=0)
"""
#根据mean画直方图
mean = mean.sort_values()
mean = mean.values
plt.figure(figsize=(10,8))
plt.bar(range(len(mean)),mean)

#plt.savefig(config.figure_save_path+'mean.png')
plt.show()
plt.close()
"""
attributions_df.to_csv('pm_attributions_n=10000.csv')
# %%
am_attributions = pd.read_csv('am_attributions_n=10000.csv',index_col=0)
pm_attributions = pd.read_csv('pm_attributions_n=10000.csv',index_col=0)
all_attributions = pd.read_csv('all_attributions_n=10000.csv',index_col=0)

def top_rows(df):
    df = df.abs()
    df = df.sort_values(by=0,axis=1,ascending=False)
    top_percent = 0.12
    top_num = int(df.shape[1]*top_percent)
    df = df.iloc[:,:top_num]
    return df

am_mean= am_attributions.mean(axis=0)
pm_mean = pm_attributions.mean(axis=0)
all_mean = all_attributions.mean(axis=0)


# %%
