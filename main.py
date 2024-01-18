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
np.random.seed(42)
import torch
torch.manual_seed(42)
from model.model_mlp import train,predict
from scipy.stats import spearmanr
#%%
#抽取100支股票调代码用
def select_stock_sample(df, total_selected=100):
    date_range = list(df.index.get_level_values(level='date').unique())
    # 创建一个空集合，用于存储每天都存在的股票
    common_stocks = set()

    for date in date_range:
        # 找出当天存在的股票
        stocks_on_date = set(df[df.index.get_level_values('date') == date].index.get_level_values(level='code'))
        
        # 如果是第一天，将当天存在的股票初始化为 common_stocks
        if not common_stocks:
            common_stocks = stocks_on_date
        else:
            # 取交集，即找出每天都存在的股票
            common_stocks = common_stocks.intersection(stocks_on_date)
    if total_selected==-1:
        selected_df = df[df.index.get_level_values('code').isin(common_stocks)] 
    else:  
        selected_df = pd.DataFrame()
        selected_stocks = ['000', '002', '600', '300']
    
        for stock_prefix in selected_stocks:
            # 获取以指定前缀开头的股票列表
            matching_stocks = [stock for stock in common_stocks if stock.startswith(stock_prefix)]
            
            # 计算抽取的数量，按照占比分配
            num_to_select = int(total_selected * len(matching_stocks) / len(common_stocks))
            # 随机抽取股票并添加到 selected_df 中
            selected_stocks = np.random.choice(matching_stocks, num_to_select, replace=False)
            selected_data = df[df.index.get_level_values('code').isin(selected_stocks)]
            selected_df = pd.concat([selected_df, selected_data])
    print(f"selected_df:{selected_df.shape}")
    return selected_df
#%%
class Config:
    random_seed = 42 
    #规定训练的参数
    valid_data_rate = 0.15

    #规定选用的特征集合。all,am,pm分别表示全天，早盘，晚盘
    selected_feature = "am"

    debug_mode = False #True为debug模式，采用随机抽取的100支股票集合
    
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

    #训练参数
    learning_rate = 0.001
    epoch = 150 #不考虑早停的前提下整个模型训练多少遍
    patience = 15
    batch_size = 1

    #训练方式（是否增量训练）
    add_train = False
    do_train = False
    do_validation = False
    do_predict = False
    shuffle_train_data = False
    use_cuda = True

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
    
    save_processed_train_data = True #如果已经有处理好的train_data,这项为false，若为True则会从头处理train data，保存到processed_train_data_path中

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

    
class Data:
    def __init__(self,config):
        self.config = config
        self.data,self.data_column_name = self.read_data()

        #排除缺失数据点在10w以上的feature
        nan_count = self.data.isna().sum()
        invalid_features = nan_count[nan_count.values>1000000].index.tolist()
        self.data = self.data.drop(columns = invalid_features)

        #去掉缺失值
        self.data = self.data.replace([np.inf,-np.inf],np.nan)
        self.data = self.data.replace(np.nan,0)
        #筛选出universe中的数据
        self.data = self.data[self.data[self.config.universe] == 1]
        #去掉涨跌停数据
        if config.limit != '':
            self.data = self.data[self.data[self.config.limit] == 0]
        #把无关的列删掉
        irrelevant_cols = self.data.columns[self.data.columns.str.startswith('u')].to_list() + self.data.columns[self.data.columns.str.startswith('y')].to_list()
        irrelevant_cols.remove(self.config.label)
        self.data = self.data.drop(columns = irrelevant_cols)

        #选用选定的特征
        if self.config.selected_feature != None:
            selected_feature = self.data.columns[self.data.columns.str.contains(self.config.selected_feature)].to_list()
            self.data = self.data.loc[:,[self.config.label]+selected_feature]

        self.normalized_data = self.data.loc[:,self.data.columns.str.startswith('x')]
        self.label = self.data.loc[:,self.config.label].to_frame()

        if config.label_normalization=='profile':
            self.label = normalization_cross_profile(self.label)
        elif config.label_normalization == "rank" :
            self.label = rank_cross_profile(self.label)
        
        self.label = self.label[self.config.label].astype(float)       

    def read_data(self):
        init_data = pd.read_hdf(self.config.train_data_path)
        return init_data,init_data.columns.tolist()
    
    def remove_outliers(self,df,columns):
        for col in columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_limit = Q1 - 1.5 * IQR
            upper_limit = Q3 + 1.5 * IQR
            df[col] = df[col].clip(lower_limit, upper_limit)
        return df
    
    def get_train_and_valid_data(self):
        #把训练集按日期划分出来
        self.normalized_data = self.normalized_data.sort_index(level=['date','code'])
        self.label = self.label.sort_index(level=['date','code'])

        feature_data = self.normalized_data.loc[:self.config.train_end_date]
        #处理label data
        label_data = self.label.loc[:self.config.train_end_date]

        print(f"splitting data...")
        
        unique_dates = feature_data.index.get_level_values('date').unique()

        train_dates,valid_dates = train_test_split(unique_dates,test_size = self.config.valid_data_rate,
                                                           random_state=self.config.random_seed, shuffle=self.config.shuffle_train_data)
        train_x_df = feature_data.loc[feature_data.index.get_level_values('date').isin(train_dates)]
        valid_x_df = feature_data.loc[feature_data.index.get_level_values('date').isin(valid_dates)]
        train_y_df = label_data.loc[label_data.index.get_level_values('date').isin(train_dates)]
        valid_y_df = label_data.loc[label_data.index.get_level_values('date').isin(valid_dates)]

        if config.feature_nomalization == 'uniform':
            self.scaler = QuantileTransformer(output_distribution='uniform')
        elif config.feature_nomalization == 'normal':
            self.scaler = QuantileTransformer(output_distribution='normal')
        elif config.feature_nomalization == 'minmax':
            self.scaler = MinMaxScaler()

        train_x_scaled = self.scaler.fit_transform(train_x_df[train_x_df.columns])
        train_x_df = pd.DataFrame(train_x_scaled, index=train_x_df.index, columns=train_x_df.columns)

        valid_x_scaled = self.scaler.transform(valid_x_df[valid_x_df.columns])
        valid_x_df = pd.DataFrame(valid_x_scaled, index=valid_x_df.index, columns=valid_x_df.columns)
       
        return train_x_df,valid_x_df,train_y_df,valid_y_df
       
    def get_test_data(self):
        feature_start_date = self.config.test_start_date
        feature_data = self.normalized_data.loc[feature_start_date:]

        #提取label data
        label_data = self.label.loc[feature_start_date:]
     
        test_X_scaled = self.scaler.transform(feature_data[feature_data.columns])
        test_X = pd.DataFrame(test_X_scaled,index = feature_data.index,columns=feature_data.columns)
        
        test_Y = label_data
        return test_X,test_Y

# 定义计算斯皮尔曼相关系数的函数
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


#%%
#测试以上函数
config = Config()
def load_train_data(path = config.processed_train_data_path):
    train_x = pd.read_hdf(os.path.join(path,'train_x.h5'))
    valid_x = pd.read_hdf(os.path.join(path,'valid_x.h5'))
    train_y = pd.read_hdf(os.path.join(path,"train_y.h5"))
    valid_y = pd.read_hdf(os.path.join(path,"valid_y.h5"))

    #print(train_y.head())
    #print(valid_y.head())
    train_and_valid_data = (train_x,valid_x,train_y,valid_y)

    return train_and_valid_data

def load_test_data(path=config.processed_test_data_path):
    test_x = pd.read_hdf(os.path.join(path,"test_x.h5"))
    test_y = pd.read_hdf(os.path.join(path,"test_y.h5"))

    return test_x,test_y

def normalization_cross_profile(df):
        """
        做截面上的normalization，固定日期，对feature在股票集上的分布做标准化
        """
        result_df = pd.DataFrame(index=df.index,columns=df.columns)
        unique_dates = df.index.get_level_values('date').unique()
        for date in unique_dates:
            date_data = df[df.index.get_level_values('date') == date]

            #对截面上的数据进行标准化
            mean = date_data.mean(axis=0)
            std = date_data.std(axis=0)
            std[std==0] = 1e-6
            normalized_date_data = (date_data-mean)/std

            result_df.loc[result_df.index.get_level_values('date')==date] = normalized_date_data
        return result_df

def rank_cross_profile(df):
    result_df = pd.DataFrame(index=df.index,columns=df.columns)
    unique_dates = df.index.get_level_values('date').unique()
    for date in unique_dates:
        date_data = df[df.index.get_level_values('date') == date]

        # 对截面上的数据进行排序
        ranked_date_data = date_data.rank(axis=0)
        mean = ranked_date_data.mean(axis=0)
        std = ranked_date_data.std(axis=0)
        std[std==0] = 1e-6
        normalize_ranked_date_data = (ranked_date_data-mean)/std
        result_df.loc[result_df.index.get_level_values('date') == date] = normalize_ranked_date_data

    return result_df


if config.save_processed_train_data == True:
    start_time = time.time()
    data_module = Data(config)
    end_time = time.time()
    print("time:",end_time-start_time)
    train_x,valid_x,train_y,valid_y = data_module.get_train_and_valid_data()

    train_x.to_hdf(config.processed_train_data_path+"train_x.h5",key='train_x',mode='w')
    valid_x.to_hdf(config.processed_train_data_path+"valid_x.h5",key="valid_x",mode='w')
    train_y.to_hdf(config.processed_train_data_path+"train_y.h5",key='train_y',mode='w')
    valid_y.to_hdf(config.processed_train_data_path+"valid_y.h5",key="valid_y",mode='w')
    print("train data saved!")
       

    test_x,test_y = data_module.get_test_data()
    print(test_x.shape,test_y.shape)
    #print(test_x.head())
    #print(test_y.head())
    test_x.to_hdf(config.processed_test_data_path+"test_x.h5",key='test_x',mode='w')
    test_y.to_hdf(config.processed_test_data_path+"test_y.h5",key="test_y",mode='w')
    print("test data saved!")
else: 
    print("Loading train data...")
    train_and_valid_data = load_train_data(config.processed_train_data_path)
    print("Loading test data...")
    test_x,test_y = load_test_data(config.processed_test_data_path)
    print("Data loaded!")
    
np.random.seed(config.random_seed)

if config.do_train:
    print(f"Training...")
    config.input_size = len(train_and_valid_data[0].columns)
    train(config,train_and_valid_data)
    print("Train finished!")


if config.do_validation:
    train_x,valid_x,train_y,valid_y = train_and_valid_data
    print(f"Use validation data to predict...")
    y_hat = predict(config,valid_x)
    save_path = './valid_result/'+f"{config.subframe}/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)    # makedirs 递归创建目录
    np.save(save_path+"model_"+config.frame,y_hat)
    print("valid result saved!")
    ic = save_test_data(y_hat,valid_y,config.valid_save_path)
    print(f"ic={ic}")

if config.do_predict:
    print(f"Testing...")
    Y_hat_test = predict(config,test_x)
    Y_hat = predict(config,train_and_valid_data[0])
    save_path = './train_result/'+f"{config.subframe}/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)    # makedirs 递归创建目录
    np.save(save_path+"model_"+config.frame,Y_hat)
    print("result saved!")
    ic = save_test_data(Y_hat_test,test_y,config.predict_save_path)
    print(f"ic={ic[0]},rank ic = {ic[1]}")
