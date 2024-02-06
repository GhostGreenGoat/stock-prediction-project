#%%
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pdb
import pickle
import gc
import scipy.stats as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from scipy.stats import spearmanr
#%%
#读取验证集数据
all_files_mse = [pd.read_csv(f'/home/laiminzhi/reconfiguration_code/predict_data/y3_label__all_normal_no_ud_limit_h2/selected_feature(all)_loss(advance)_depth4_sizes(249_128_128_1)_bd_valid/{f}',
                         dtype={'date':str})for f in sorted(os.listdir('/home/laiminzhi/reconfiguration_code/predict_data/y3_label__all_normal_no_ud_limit_h2/selected_feature(all)_loss(advance)_depth4_sizes(249_128_128_1)_bd_valid/'))]
all_files_wmse1 = [pd.read_csv(f'/home/laiminzhi/reconfiguration_code/predict_data/y3_label__all_normal_no_ud_limit_h2/selected_feature(all)_loss(wmse)_depth4_sizes(249_128_128_1)_bd_valid/{f}',
                         dtype={'date':str})for f in sorted(os.listdir('/home/laiminzhi/reconfiguration_code/predict_data/y3_label__all_normal_no_ud_limit_h2/selected_feature(all)_loss(wmse)_depth4_sizes(249_128_128_1)_bd_valid/'))]

all_files_profile = [pd.read_csv(f'/home/laiminzhi/reconfiguration_code/predict_data/y3_label_profile_all_normal_no_ud_limit_h2/selected_feature(all)_loss(combine)_depth4_sizes(249_128_128_1)_bd_valid/{f}',
                         dtype={'date':str})for f in sorted(os.listdir('/home/laiminzhi/reconfiguration_code/predict_data/y3_label_profile_all_normal_no_ud_limit_h2/selected_feature(all)_loss(combine)_depth4_sizes(249_128_128_1)_bd_valid/'))]
all_files_rank = [pd.read_csv(f'/home/laiminzhi/reconfiguration_code/predict_data/y3_label_rank_all_normal_no_ud_limit_h2/selected_feature(all)_loss(combine)_depth4_sizes(249_128_128_1)_bd_valid/{f}',
                         dtype={'date':str})for f in sorted(os.listdir('/home/laiminzhi/reconfiguration_code/predict_data/y3_label_rank_all_normal_no_ud_limit_h2/selected_feature(all)_loss(combine)_depth4_sizes(249_128_128_1)_bd_valid/'))]

all_files_wmse2 = [pd.read_csv(f'/home/laiminzhi/reconfiguration_code/predict_data/y3_label__all_normal_no_ud_limit_h2/selected_feature(all)_loss(wmse_decay)_depth4_sizes(249_128_128_1)_bd_valid/{f}',
                         dtype={'date':str})for f in sorted(os.listdir('/home/laiminzhi/reconfiguration_code/predict_data/y3_label__all_normal_no_ud_limit_h2/selected_feature(all)_loss(wmse_decay)_depth4_sizes(249_128_128_1)_bd_valid/'))]

all_files_quantile = [pd.read_csv(f'/home/laiminzhi/reconfiguration_code/predict_data/y3_label__all_normal_no_ud_limit_h2/selected_feature(all)_loss(quantile)_depth4_sizes(249_128_128_1)_bd_valid/{f}',
                         dtype={'date':str})for f in sorted(os.listdir('/home/laiminzhi/reconfiguration_code/predict_data/y3_label__all_normal_no_ud_limit_h2/selected_feature(all)_loss(log)_depth4_sizes(249_128_128_1)_bd_valid/'))]
all_files_log =  [pd.read_csv(f'/home/laiminzhi/reconfiguration_code/predict_data/y3_label__all_normal_no_ud_limit_h2/selected_feature(all)_loss(log)_depth4_sizes(249_128_128_1)_bd_valid/{f}',
                         dtype={'date':str})for f in sorted(os.listdir('/home/laiminzhi/reconfiguration_code/predict_data/y3_label__all_normal_no_ud_limit_h2/selected_feature(all)_loss(log)_depth4_sizes(249_128_128_1)_bd_valid/'))]
#%%
def get_predict_data(path):
    # 读取预测数据
    # path: 预测数据的路径
    # return: 预测数据的dataframe
    data = pd.read_csv(path)
    data['Y_rank'] = data['Y'].rank(method='min')
    data['Y_hat_rank'] = data['Y_hat'].rank(method='min')
    return data


def draw_pdf(pred,label):
    vector1 = pred
    vector2 = label

    #vector1 = (vector1 - vector1.min()) / (vector1.max() - vector1.min())
    #vector2 = (vector2 - vector2.min()) / (vector2.max() - vector2.min())


    # 计算每个向量的PDF
    kde1 = gaussian_kde(vector1)
    kde2 = gaussian_kde(vector2)

    # 设置绘图的点
    x_min = min(vector1.min(), vector2.min())
    x_max = max(vector1.max(), vector2.max())
    x = np.linspace(x_min, x_max, 1000)

    # 绘制两个向量的PDF
    plt.figure(figsize=(8, 6))
    plt.plot(x, kde1(x), label='pred PDF')
    plt.plot(x, kde2(x), label='label PDF')

    # 添加图例
    plt.legend()

    # 显示图
    plt.show()


#绘制Y和Y_hat的散点图
def draw_scatter(pred,label):
    # 假设你有两个向量
    vector1 = pred
    vector2 = label

    #vector1 = (vector1 - vector1.min()) / (vector1.max() - vector1.min())
    #vector2 = (vector2 - vector2.min()) / (vector2.max() - vector2.min())

    # 创建一个带有2个子图的图形，它们并排排列
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # 在第一个子图中绘制vector1
    axs[0].scatter(range(len(vector1)), vector1, color='b', label='pred')
    axs[0].set_title('pred')
    axs[0].set_xlabel('code')
    axs[0].set_ylabel('Y_hat')
    axs[0].legend()

    # 在第二个子图中绘制vector2
    axs[1].scatter(range(len(vector2)), vector2, color='r', label='label')
    axs[1].set_title('label')
    axs[1].set_xlabel('code')
    axs[1].set_ylabel('Y')
    axs[1].legend()

    # 显示图形
    plt.show()

def quantile_ic(df,rank_ic_per_quantile,object = 'Y'):
    # 计算真实回报率的分位数
    quantile_labels = range(1, 6)
    df['quantile'] = pd.qcut(df['Y'], 5, labels=quantile_labels)

    for quantile in quantile_labels:
        # 选取当前分位数组内的数据
        subset = df[df['quantile'] == quantile]
        
        # 计算真实值和预测值的秩次
        true_ranks = subset['Y'].rank()
        predicted_ranks = subset['Y_hat'].rank()
        
        # 计算秩次的相关系数（Rank IC）
        rank_ic = spearmanr(true_ranks, predicted_ranks).correlation
        rank_ic_per_quantile[quantile].append(rank_ic)

    return rank_ic_per_quantile

#获取截面return的quantile，截取特定的quantile的股票
def get_specific_quantile(df,quantile_num,quantile):
    quantile_labels = range(1, quantile_num+1)
    df['quantile'] = pd.qcut(df['Y_hat'], quantile_num, labels=quantile_labels)
    subset = df[df['quantile'] == quantile]
    return subset

#画出每个quantile的rank_ic
def draw_quantile(all_files,object = 'Y'):
    rank_ic_per_quantile = {i: [] for i in range(1, 6)}
    for file in all_files:
        rank_ic_per_quantile = quantile_ic(file,rank_ic_per_quantile)


    #把rank_ic_per_quantile中每个quantile的向量按照向量值为y，向量index为x画在同一张图上
    for quantile in rank_ic_per_quantile:
        y = rank_ic_per_quantile[quantile]
        #平滑y，取5天均值，y为list
        y = pd.Series(y).rolling(5).mean().to_list()
        x = range(len(y))
        plt.plot(x,y
                ,label=f'quantile{quantile}')
        #规定label
        plt.xlabel('date')
        plt.ylabel('rank_ic')
        plt.title('rank_ic_per_quantile')
        plt.legend()

    plt.show()  
# %%
#循环查看一年内每个截面上的预测值和真实值的分布
for i,file in enumerate(all_files_quantile):
    draw_pdf(file['Y_hat'],file['Y'])
    draw_scatter(file['Y_hat'],file['Y'])
    break
#%%
draw_quantile(all_files_quantile,object = 'Y')
# %%
#读取训练数据
train_result = np.load('/home/laiminzhi/reconfiguration_code/train_result/y3_label__all_normal_no_ud_limit_h2/model_selected_feature(all)_loss(advance)_depth4_sizes(249_128_128_1)_bd.npy')
train_y = pd.read_hdf('/home/laiminzhi/reconfiguration_code/train_data/y3_label__all_normal_no_ud_limit_h2/train_y.h5').to_frame().rename(columns={'y3_label':'Y'})
#把train_y和train_result合并
train_y['Y_hat'] = train_result
grouped_train_y = train_y.groupby('date')
#%%
date_list = train_y.index.get_level_values('date').unique().to_list()
#从date_list中随机抽取100个日期
sample_date = np.random.choice(date_list,100)
#画sample_list中的日期的pdf
for date in sample_date:
    draw_pdf(grouped_train_y.get_group(date)['Y_hat'],grouped_train_y.get_group(date)['Y'])

#%%
#画所有日期上的分布
draw_pdf(train_y['Y_hat'],train_y['Y'])
# %%
#计算pred和label的残差的均值
residual = train_y['Y_hat'] - train_y['Y']
residual_mean = residual.mean()
# %%
#把residual画出来
plt.scatter(range(len(residual[1000:2000])),residual[1000:2000])
# %%
draw_quantile(all_files_mse,object = 'Y')
draw_quantile(all_files_wmse1,object = 'Y')
draw_quantile(all_files_profile,object = 'Y')
draw_quantile(all_files_rank,object = 'Y')

draw_quantile(all_files_mse,object = 'Y_hat')
draw_quantile(all_files_wmse1,object = 'Y_hat')
draw_quantile(all_files_profile,object = 'Y_hat')
draw_quantile(all_files_rank,object = 'Y_hat')

#%%
draw_quantile(all_files_wmse2,object = 'Y')
draw_quantile(all_files_wmse2,object = 'Y_hat')

#%%
#读入测试集数据
all_files_mse_test = [pd.read_csv(f'/home/laiminzhi/reconfiguration_code/predict_data/y3_label__all_normal_no_ud_limit_h2/selected_feature(all)_loss(advance)_depth4_sizes(249_128_128_1)_bd/{f}',
                         dtype={'date':str})for f in sorted(os.listdir('/home/laiminzhi/reconfiguration_code/predict_data/y3_label__all_normal_no_ud_limit_h2/selected_feature(all)_loss(advance)_depth4_sizes(249_128_128_1)_bd/'))]
all_files_wmse1_test = [pd.read_csv(f'/home/laiminzhi/reconfiguration_code/predict_data/y3_label__all_normal_no_ud_limit_h2/selected_feature(all)_loss(wmse)_depth4_sizes(249_128_128_1)_bd/{f}',
                         dtype={'date':str})for f in sorted(os.listdir('/home/laiminzhi/reconfiguration_code/predict_data/y3_label__all_normal_no_ud_limit_h2/selected_feature(all)_loss(wmse)_depth4_sizes(249_128_128_1)_bd/'))]

all_files_profile_test = [pd.read_csv(f'/home/laiminzhi/reconfiguration_code/predict_data/y3_label_profile_all_normal_no_ud_limit_h2/selected_feature(all)_loss(combine)_depth4_sizes(249_128_128_1)_bd/{f}',
                         dtype={'date':str})for f in sorted(os.listdir('/home/laiminzhi/reconfiguration_code/predict_data/y3_label_profile_all_normal_no_ud_limit_h2/selected_feature(all)_loss(combine)_depth4_sizes(249_128_128_1)_bd/'))]
all_files_rank_test = [pd.read_csv(f'/home/laiminzhi/reconfiguration_code/predict_data/y3_label_rank_all_normal_no_ud_limit_h2/selected_feature(all)_loss(combine)_depth4_sizes(249_128_128_1)_bd/{f}',
                         dtype={'date':str})for f in sorted(os.listdir('/home/laiminzhi/reconfiguration_code/predict_data/y3_label_rank_all_normal_no_ud_limit_h2/selected_feature(all)_loss(combine)_depth4_sizes(249_128_128_1)_bd/'))]

all_files_wmse2_test = [pd.read_csv(f'/home/laiminzhi/reconfiguration_code/predict_data/y3_label__all_normal_no_ud_limit_h2/selected_feature(all)_loss(wmse_decay)_depth4_sizes(249_128_128_1)_bd/{f}',
                         dtype={'date':str})for f in sorted(os.listdir('/home/laiminzhi/reconfiguration_code/predict_data/y3_label__all_normal_no_ud_limit_h2/selected_feature(all)_loss(wmse_decay)_depth4_sizes(249_128_128_1)_bd/'))]

all_files_quantile_test = [pd.read_csv(f'/home/laiminzhi/reconfiguration_code/predict_data/y3_label__all_normal_no_ud_limit_h2/selected_feature(all)_loss(quantile)_depth4_sizes(249_128_128_1)_bd/{f}',
                         dtype={'date':str})for f in sorted(os.listdir('/home/laiminzhi/reconfiguration_code/predict_data/y3_label__all_normal_no_ud_limit_h2/selected_feature(all)_loss(quantile)_depth4_sizes(249_128_128_1)_bd/'))]

all_files_log_test =  [pd.read_csv(f'/home/laiminzhi/reconfiguration_code/predict_data/y3_label__all_normal_no_ud_limit_h2/selected_feature(all)_loss(log)_depth4_sizes(249_128_128_1)_bd/{f}',
                            dtype={'date':str})for f in sorted(os.listdir('/home/laiminzhi/reconfiguration_code/predict_data/y3_label__all_normal_no_ud_limit_h2/selected_feature(all)_loss(log)_depth4_sizes(249_128_128_1)_bd/'))]
# %%
def get_qantile_ratio(df,quantile=5,quantile_num=5):
    df['quantile_y'] = pd.qcut(df['Y'], quantile_num, labels=range(1, quantile_num+1))
    df['quantile_y_hat'] = pd.qcut(df['Y_hat'], quantile_num, labels=range(1, quantile_num+1))
    subset = df[df['quantile_y_hat'] == quantile]
    #查看subset中quantile_y也为quantile的比例
    quantile_y_true = subset[subset['quantile_y'] == quantile]
    quantile_y_true_ratio = len(quantile_y_true)/len(subset)       

    return quantile_y_true_ratio

def get_mistake_quantile_ratio(df,quantile = 5,quantile_num=5):
    df['quantile_y'] = pd.qcut(df['Y'], quantile_num, labels=range(1, quantile_num+1))
    df['quantile_y_hat'] = pd.qcut(df['Y_hat'], quantile_num, labels=range(1, quantile_num+1))
    subset = df[df['quantile_y_hat'] == quantile]
    #查看subset中quantile_y为中位数以后的quantiel的比例
    quantile_y_mistake = subset[subset['quantile_y'] <= quantile_num//2]
    quantile_y_mistake_ratio = len(quantile_y_mistake)/len(subset)
    return quantile_y_mistake_ratio

def get_negative_return_ratio(df,quantile=5,quantile_num=5):
    df['quantile_y'] = pd.qcut(df['Y'], quantile_num, labels=range(1, quantile_num+1))
    df['quantile_y_hat'] = pd.qcut(df['Y_hat'], quantile_num, labels=range(1, quantile_num+1))
    subset = df[df['quantile_y_hat'] == quantile]
    #查看subset中Y为负的比例
    quantile_y_mistake = subset[subset['Y'] <= 0]
    quantile_y_mistake_ratio = len(quantile_y_mistake)/len(subset)
    return quantile_y_mistake_ratio

def get_positive_return_ratio(df,quantile=5,quantile_num=5):
    df['quantile_y'] = pd.qcut(df['Y'], quantile_num, labels=range(1, quantile_num+1))
    df['quantile_y_hat'] = pd.qcut(df['Y_hat'], quantile_num, labels=range(1, quantile_num+1))
    subset = df[df['quantile_y_hat'] == quantile]
    #查看subset中Y为正的比例
    quantile_y_mistake = subset[subset['Y'] > 0]
    quantile_y_mistake_ratio = len(quantile_y_mistake)/len(subset)
    return quantile_y_mistake_ratio

#%%
quantile_ratio_dict = {i: [] for i in range(1, 6)}
for file in all_files_mse:
    for q in range(1,6):
        quantile_y_true_ratio = get_qantile_ratio(file,quantile=q)
        quantile_ratio_dict[q].append(quantile_y_true_ratio)

#把rank_ic_per_quantile中每个quantile的向量按照向量值为y，向量index为x画在同一张图上
for quantile in quantile_ratio_dict:
    y = quantile_ratio_dict[quantile]
    #平滑y，取5天均值，y为list
    y = pd.Series(y).rolling(5).mean().to_list()
    x = range(len(y))
    plt.plot(x,y
            ,label=f'quantile{quantile}')
    #规定label
    plt.xlabel('date')
    plt.ylabel('correct_y_quantile_ratio')
    plt.title('correct_y')
    plt.legend()

plt.show()  


# %%
all_file = {'mse':all_files_mse,'wmse1':all_files_wmse1,'profile':all_files_profile,'rank':all_files_rank,'wmse2':all_files_wmse2,'quantile':all_files_quantile,'log':all_files_log}
all_file_test = {'mse':all_files_mse_test,'wmse1':all_files_wmse1_test,'profile':all_files_profile_test,'rank':all_files_rank_test,'wmse2':all_files_wmse2_test,'quantile':all_files_quantile_test,'log':all_files_log_test}
#%%
mistake_quatile_ratio = dict()
#获取quantile5中mistake quantile的比例
for name,all_files in all_file.items():
    mistake_quatile_ratio[name] = []
    for file in all_files:
        mistake_quatile_ratio[name].append( get_negative_return_ratio(file))

    y = mistake_quatile_ratio[name]
    #平滑y，取5天均值，y为list
    y = pd.Series(y).rolling(5).mean().to_list()
    x = range(len(y))
    plt.plot(x,y,label=name)
    #规定label
    plt.xlabel('date')
    plt.ylabel('mistake_y_quantile_ratio')
    plt.title('mistake_y')
    plt.legend()

plt.show()  

# %%
#获取正确quantile的比例
correct_quatile_ratio = dict()
#获取quantile5中mistake quantile的比例
for name,all_files in all_file_test.items():
    correct_quatile_ratio[name] = []
    for file in all_files:
        correct_quatile_ratio[name].append( get_positive_return_ratio(file))

    y = correct_quatile_ratio[name]
    #平滑y，取5天均值，y为list
    y = pd.Series(y).rolling(5).mean().to_list()
    x = range(len(y))
    plt.plot(x,y,label=name)
    #规定label
    plt.xlabel('date')
    plt.ylabel('correct_y_quantile_ratio')
    plt.title('correct_y')
    plt.legend()

plt.show()  
# %%
#查看all_files_rank中y_hat前10%的股票的y的quantile和y_hat的quantile的关系
correct_rate = dict()
for name,all_files in all_file_test.items():
    correct_rate[name] = []
    for file in all_files:
        x = file
        x['quantile_y'] = pd.qcut(file['Y'], 5, labels=range(1, 6))
        x['quantile_y_hat'] = pd.qcut(file['Y_hat'], 5, labels=range(1, 6))
        #x选出y_hat前10%的股票
        x = file[file['Y_hat'] >= file['Y_hat'].quantile(0.9)]
        #给x中的每个股票标记y_hat的quantile和y的quantile
        subset = x[x['quantile_y_hat'] == 5]
        #查看subset中quantile_y也为quantile的比例
        quantile_y_true = subset[subset['quantile_y'] == 5]
        quantile_y_true_ratio = len(quantile_y_true)/len(subset) 
        correct_rate[name].append(quantile_y_true_ratio)

    y = correct_rate[name]
    y = pd.Series(y).rolling(5).mean().to_list()
    x = range(len(y))
    plt.plot(x,y,label=name)
    plt.legend()
    plt.title("90% quantile correct rate")


# %%
#把correct rate和错误rate画在一张图上
all_rate = dict()
for name,all_files in all_file.items():
    correct_name = name+'_correct'
    mistake_name = name+'_mistake'
    all_rate[correct_name] = []
    all_rate[mistake_name] = []
    for file in all_files:
        x=file
        x['quantile_y'] = pd.qcut(file['Y'], 5, labels=range(1, 6))
        x['quantile_y_hat'] = pd.qcut(file['Y_hat'], 5, labels=range(1, 6))
        #x选出y_hat前10%的股票
        x = file[file['Y_hat'] >= file['Y_hat'].quantile(0.9)]
        subset=x
        #查看subset中quantile_y也为quantile的比例
        quantile_y_true = subset[subset['quantile_y'] > 3]
        quantile_y_true_ratio = len(quantile_y_true)/len(subset)
        all_rate[correct_name].append(quantile_y_true_ratio)

        #查看subset中quantile_y为中位数以后的quantiel的比例
        quantile_y_mistake = subset[subset['quantile_y'] < 3]
        quantile_y_mistake_ratio = len(quantile_y_mistake)/len(subset)
        all_rate[mistake_name].append(quantile_y_mistake_ratio)

    y1 = all_rate[correct_name]
    y2 = all_rate[mistake_name]
    y1 = pd.Series(y1).rolling(30).mean().to_list()
    y2 = pd.Series(y2).rolling(30).mean().to_list()
    x = range(len(y1))
    plt.plot(x,y1,label=correct_name)
    #画虚线
    plt.plot(x,y2,label=mistake_name,linestyle='--')
    #把ldegend放到图外面
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

    plt.title("90% quantile correct and mistake rate")




# %%
#画出预测正确-预测错误的比例
all_rate = dict()
for name,all_files in all_file_test.items():
    correct_name = name+'_correct'
    mistake_name = name+'_mistake'
    all_rate[correct_name] = []
    all_rate[mistake_name] = []
    for file in all_files:
        x=file
        x['quantile_y'] = pd.qcut(file['Y'], 5, labels=range(1, 6))
        x['quantile_y_hat'] = pd.qcut(file['Y_hat'], 5, labels=range(1, 6))
        #x选出y_hat前10%的股票
        x = file[file['Y_hat'] > file['Y_hat'].quantile(0.9)]
        subset=x
        #查看subset中quantile_y也为quantile的比例
        quantile_y_true = subset[subset['Y'] > file['Y'].quantile(0.9) ]
        quantile_y_true_ratio = len(quantile_y_true)/len(subset)
        all_rate[correct_name].append(quantile_y_true_ratio)

        #查看subset中quantile_y为中位数以后的quantiel的比例
        quantile_y_mistake = subset[subset['Y'] < file['Y'].quantile(0.1) ]
        quantile_y_mistake_ratio = len(quantile_y_mistake)/len(subset)
        all_rate[mistake_name].append(quantile_y_mistake_ratio)

    y1 = all_rate[correct_name]
    y2 = all_rate[mistake_name]
    y1 = pd.Series(y1).rolling(30).mean()
    y2 = pd.Series(y2).rolling(30).mean()
    x = range(len(y1))
    plt.plot(x,(y1-y2).to_list(),label=name)
    #把ldegend放到图外面
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

    plt.title("N(Y>90%)-N(Y<10%),N(Y_hat>90%)")

# %%
#计算精确率(y_hat大于0.9的部分中y也大于0.9的占比)
#计算召回率(y>0.9的部分中y_hat也大于0.9的占比)
all_rate = dict()
for name,all_files in all_file.items():
    precision_name = name+'_precision'
    recall_name = name+'_recall'
    all_rate[precision_name] = []
    all_rate[recall_name] = []
    for file in all_files:
        x=file
        x['quantile_y'] = pd.qcut(file['Y'], 5, labels=range(1, 6))
        x['quantile_y_hat'] = pd.qcut(file['Y_hat'], 5, labels=range(1, 6))
        #x选出y_hat前10%的股票
        y_hat_true = file[file['Y_hat'] >= file['Y_hat'].quantile(0.9)]
        y_true = file[file['Y'] >= file['Y'].quantile(0.9)]

        #查看y_hat_true中Y也大于0.9的比例
        precision = len(y_hat_true[y_hat_true['Y'] >= file['Y'].quantile(0.9)])/len(y_hat_true)
        #查看y_true中y_hat也大于0.9的比例
        recall = len(y_true[y_true['Y_hat'] >= file['Y_hat'].quantile(0.9)])/len(y_true)

        all_rate[precision_name].append(precision)
        all_rate[recall_name].append(recall)

    y1 = all_rate[precision_name]
    y2 = all_rate[recall_name]
    y1 = pd.Series(y1).rolling(30).mean().to_list()
    y2 = pd.Series(y2).rolling(30).mean().to_list()
    x = range(len(y1))
    plt.plot(x,y1,label=precision_name)
    #画虚线
    #plt.plot(x,y2,label=recall_name,linestyle='--')
    #把ldegend放到图外面
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

    plt.title("90% quantile correct and mistake rate")
 
# %%
#计算alpha
#读取原始数据
xy = pd.read_hdf('/home/laiminzhi/wenbin/DL_stock_combo/data/xy_data/xy_data_new.h5').reset_index()
#%%
#获取截面return的quantile，截取特定的quantile的股票
def get_specific_quantile(df,quantile_num,quantile):
    quantile_labels = range(1, quantile_num+1)
    df['quantile'] = pd.qcut(df['Y_hat'], quantile_num, labels=quantile_labels)
    subset = df[df['quantile'] == quantile]
    return subset

quantiles=10
yest_dict = {i: pd.DataFrame() for i in range(1, quantiles+1)}       

for file in all_files_log:
    for q in range(1,quantiles+1):
        subset = get_specific_quantile(file,quantile_num=quantiles,quantile=q)
        yest_dict[q]=pd.concat([yest_dict[q],subset],axis=0)

#yest = pd.concat(all_files_wmse2, axis=0) #贴合所有的预测值

def get_XY(yest):
    universe = 'univ_tradable'

    XY = xy.loc[xy[universe]==1,:'ud_limit_h4']
    XY= XY.rename(columns={'y1':'y'})
    XY = pd.merge(XY, yest,on=['date','code'],how='inner')

    ##---- 1. benchmark ----##
    XY['yest'] = XY['Y_hat']
    return XY

def cal_alpha(XY):
    #查看pnl如何计算
    d1 = XY.copy()
    enterRatio = 0
    exitRatio = 0
    ## 1) calculate yestRank;
    d1['yestRank'] = d1.groupby('date')['yest'].rank(method='average',na_option='keep',ascending=True,pct=True)
    rtnMat = pd.pivot_table(data=d1,index='date',columns='code',values='y',dropna=False)
    yestMat = pd.pivot_table(data=d1,index='date',columns='code',values='yest',dropna=False)
    yestRankMat = pd.pivot_table(data=d1,index='date',columns='code',values='yestRank',dropna=False).fillna(0)
    posiMat = pd.DataFrame(np.full(yestRankMat.shape,fill_value=0),index=yestRankMat.index,columns=yestRankMat.columns)
    ud_limitMat = pd.pivot_table(data=d1,index='date',columns='code',values='ud_limit_h2',dropna=False).fillna(0)

    ## 2) calPosiMat： ## no buy if up_limit && no sell if down_limit;
    for i,row_index in enumerate(posiMat.index):
        if (i==0):
            continue
        flag1 = (yestRankMat.iloc[i,:]>enterRatio)
        flag2 = (posiMat.iloc[i-1,:]==0) & (ud_limitMat.iloc[i,:]==1)
        posiMat.loc[row_index,(~flag2 & flag1)] = 1

        flag3 = (yestRankMat.iloc[i,:]>exitRatio) & (yestRankMat.iloc[i,:]<=enterRatio)
        flag4 = (posiMat.iloc[i-1,:]==1)
        posiMat.loc[row_index,(flag3 & flag4)] = 1

        flag5 = (posiMat.iloc[i-1,:]==1) & (posiMat.iloc[i,:]==0) & (ud_limitMat.iloc[i,:]==-1)
        posiMat.loc[row_index,flag5] = 1
        
        if (i== (posiMat.shape[0]-1)):## position=0 if yest=NA on last day;
            flag6 = yestMat.iloc[i,:].isna()
            posiMat.loc[row_index,flag6] = 0


    pnlMat = rtnMat * posiMat
    pnlVec = pnlMat.sum(axis=1)/(posiMat==1).sum(axis=1)
    alpha = pnlVec.mean()*1e4
    return alpha
#%%
#计算特定exit-enter ratio下的alpha
def cal_alpha(XY):
    #查看pnl如何计算
    d1 = XY.copy()
    enterRatio = 0.7
    exitRatio = 0.7
    ## 1) calculate yestRank;
    d1['yestRank'] = d1.groupby('date')['yest'].rank(method='average',na_option='keep',ascending=True,pct=True)
    rtnMat = pd.pivot_table(data=d1,index='date',columns='code',values='y',dropna=False)
    yestMat = pd.pivot_table(data=d1,index='date',columns='code',values='yest',dropna=False)
    yestRankMat = pd.pivot_table(data=d1,index='date',columns='code',values='yestRank',dropna=False).fillna(0)
    posiMat = pd.DataFrame(np.full(yestRankMat.shape,fill_value=0),index=yestRankMat.index,columns=yestRankMat.columns)
    ud_limitMat = pd.pivot_table(data=d1,index='date',columns='code',values='ud_limit_h2',dropna=False).fillna(0)

    ## 2) calPosiMat： ## no buy if up_limit && no sell if down_limit;
    for i,row_index in enumerate(posiMat.index):
        if (i==0):
            continue
        flag1 = (yestRankMat.iloc[i,:]>enterRatio)
        flag2 = (posiMat.iloc[i-1,:]==0) & (ud_limitMat.iloc[i,:]==1)
        posiMat.loc[row_index,(~flag2 & flag1)] = 1

        flag3 = (yestRankMat.iloc[i,:]>exitRatio) & (yestRankMat.iloc[i,:]<=enterRatio)
        flag4 = (posiMat.iloc[i-1,:]==1)
        posiMat.loc[row_index,(flag3 & flag4)] = 1

        flag5 = (posiMat.iloc[i-1,:]==1) & (posiMat.iloc[i,:]==0) & (ud_limitMat.iloc[i,:]==-1)
        posiMat.loc[row_index,flag5] = 1
        
        if (i== (posiMat.shape[0]-1)):## position=0 if yest=NA on last day;
            flag6 = yestMat.iloc[i,:].isna()
            posiMat.loc[row_index,flag6] = 0


    pnlMat = rtnMat * posiMat
    pnlVec = pnlMat.sum(axis=1)/(posiMat==1).sum(axis=1)
    alpha = pnlVec.mean()*1e4
    return alpha

yest = pd.concat(all_files_wmse1, axis=0) #贴合所有的预测值
XY = get_XY(yest)
alpha = cal_alpha(XY)
print(alpha)
#%%
# 计算每个quantile的alpha
q_list = []
alpha_list = []
for q,yest in yest_dict.items():
    XY = get_XY(yest)
    alpha = cal_alpha(XY)
    print(q,alpha)
    q_list.append(q)
    alpha_list.append(alpha)

#以alpha_list为y，q_list为x画图
plt.plot(q_list,alpha_list)
plt.xlabel('quantile')
plt.ylabel('alpha')
plt.title('alpha-quantile')
plt.show()
# %%
