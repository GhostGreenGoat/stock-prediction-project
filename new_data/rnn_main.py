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
from model.model_lstm import train,predict
from scipy.stats import spearmanr
#torch.backends.cudnn.enabled = False
import os
#os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
#torch.cuda.empty_cache()
#%%
class Config:
    random_seed = 42

    #规定训练的参数
    valid_data_rate = 0.15
    windowsize = 30 #用时序类模型时候的回看周期（也是时序normalization的窗口）

    #规定选用的特征集合。all,am,pm分别表示全天，早盘，晚盘
    selected_feature = "new"
    selected_feature_set = np.load('slected_feature_indices.npy')
    
    #训练集结束的日期和测试集开始的日期
    train_end_date = '20211231'
    test_start_date = '20220104'
    
    #规定训练标签,股票池
    label = 'y3_label'
    universe = 'univ_tradable'
    limit = 'ud_limit_h2' #去掉的涨跌停数据

    #规定预测未来几天
    predict_day = 1

    #规定特征和标签的处理方法
    label_normalization = '' #取值有profile,rank和空值
    feature_nomalization = 'uniform' #取值有uniform,normal,minmax,robust和空值

    #模型参数
    input_size = 71
    hidden_size = 128
    lstm_layers = 2
    dropout_rate = 0.2
    output_size = 1

    #训练参数
    learning_rate = 0.001
    epoch = 50 #不考虑早停的前提下整个模型训练多少遍
    patience = 10
    batch_size = 1

    
    #训练方式（是否增量训练）
    add_train = False
    do_train = True
    do_validation = True
    do_predict = True
    shuffle_train_data = False
    use_cuda = True
    save_processed_train_data = False #如果已经有处理好的train_data,这项为false，若为True则会从头处理train data，保存到processed_train_data_path中

    #命名参数
    loss = "mse"
    model_type = "gru"
    data = "xy_data_2836"
    
    subframe = f"{model_type}_{windowsize}_{label}_{label_normalization}_{selected_feature}_{feature_nomalization}_no_{limit}"
    frame = f"selected_feature({selected_feature}{len(selected_feature_set)})_loss({loss})_timestep({windowsize})_layers({lstm_layers})_hidden({hidden_size})_dropout({dropout_rate})"
    #路径参数
    train_data_path = f"/home/laiminzhi/data/xydata/{data}.h5"
    model_save_path = f"/home/laiminzhi/new_data/model/{subframe}/"
    figure_save_path = f"/home/laiminzhi/new_data/figure/{subframe}/"
    predict_save_path = f"/home/laiminzhi/new_data/predict_data/{data}/{subframe}/{frame}/"
    valid_save_path = f"/home/laiminzhi/new_data/predict_data/{data}/{subframe}/{frame}_valid/"
    processed_train_data_path = f"/home/laiminzhi/new_data/train_data/{data}/"+ subframe +"/"
    processed_test_data_path = f"/home/laiminzhi/new_data/test_data/{data}/"+ subframe+"/"
    
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
    print("make dir done！")
    #名称
    model_name = "model_"+frame + ".pth"



class Data:
    def __init__(self,config):
        self.config = config
        self.data,self.data_column_name = self.read_data()
        self.data.index = self.data.index.set_names(['code1','date','code'], level=[0, 1,2])
        # 删除不需要的索引层
        self.data = self.data.droplevel('code1')

        #排除缺失数据点在10w以上的feature
        nan_count = self.data.isna().sum()
        invalid_features = nan_count[nan_count.values>1000000].index.tolist()
        self.data = self.data.drop(columns = invalid_features)

        #去掉缺失值
        self.data = self.data.replace([np.inf,-np.inf],np.nan)
        self.data = self.data.replace(np.nan,0)
        
        #把无关的列删掉
        irrelevant_cols = self.data.columns[self.data.columns.str.startswith('u')].to_list() + self.data.columns[self.data.columns.str.startswith('y')].to_list()
        irrelevant_cols.remove(self.config.label)
        self.data = self.data.drop(columns = irrelevant_cols)

        self.normalized_data = self.data.loc[:,self.data.columns.str.startswith('x')]
        self.label = self.data.loc[:,self.config.label].to_frame()

        if config.label_normalization=='profile':
            self.label = normalization_cross_profile(self.label)
        elif config.label_normalization == "rank" :
            self.label = rank_cross_profile(self.label)
        
        self.label = self.label[self.config.label].astype(float) 
        self.limits = {} #用来存储每列的上下界 
        
    def read_data(self):
        init_data = pd.read_hdf(self.config.train_data_path)
        #选中需要的特征
        if self.config.selected_feature != None:
            if self.config.selected_feature == 'new':
                selected_feature_all = init_data.columns.to_list()
                selected_feature = init_data.columns[init_data.columns.str.startswith('x')].to_list()
            else:
                selected_feature = init_data.columns[init_data.columns.str.contains(self.config.selected_feature)].to_list()
                selected_feature = [col for col in selected_feature if col.startswith('x')]
                other_feature = init_data.columns[~init_data.columns.str.startswith('x')].to_list()
                selected_feature_all = other_feature + selected_feature
            init_data = init_data.loc[:,selected_feature_all]
            data_column_name = selected_feature_all
        print("init_data:",init_data.shape)
    
        return init_data,data_column_name
    
    def fit_remove_outliers(self,df,columns):
        #用来在训练集上去除极端值，记录上下界
        for col in columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_limit = Q1 - 1.5 * IQR
            upper_limit = Q3 + 1.5 * IQR
            df.loc[:,col] = df[col].clip(lower_limit, upper_limit)

            #存储每列的限制
            self.limits[col] = {'lower_limit': lower_limit, 'upper_limit': upper_limit}
        return df
    
    def transform_remove_outliers(self,df,columns):
        #在测试集和验证集上应用
        for col in columns:
            df.loc[:,col] = df[col].clip(self.limits[col]['lower_limit'], self.limits[col]['upper_limit'])
        return df
    
    def transpose_df2array(self,df):
        #用pivot_table把df转变为numpy数组
        result_data = []
        for col in df.columns:
            feature_table = df.pivot_table(index='date',columns='code',values=col)
            result_data.append(feature_table.values)

        result_data = np.array(result_data) #(243,1215,2836)
        return result_data
    


    def get_train_and_valid_data(self,get_label_index = False):
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
        elif config.feature_nomalization == 'robust':
            train_x_df = self.fit_remove_outliers(train_x_df,train_x_df.columns)
            valid_x_df = self.transform_remove_outliers(valid_x_df,valid_x_df.columns)
            self.scaler = MinMaxScaler()

        train_x_scaled = self.scaler.fit_transform(train_x_df[train_x_df.columns])
        train_x_df = pd.DataFrame(train_x_scaled, index=train_x_df.index, columns=train_x_df.columns)

        valid_x_scaled = self.scaler.transform(valid_x_df[valid_x_df.columns])
        valid_x_df = pd.DataFrame(valid_x_scaled, index=valid_x_df.index, columns=valid_x_df.columns)

        
        #获取训练数据和验证数据的numpy数据
        print("transpoing train data...")
        train_x = self.transpose_df2array(train_x_df)
        valid_x = self.transpose_df2array(valid_x_df)
        train_y = self.transpose_df2array(train_y_df.to_frame())
        valid_y = self.transpose_df2array(valid_y_df.to_frame())

        #把train_y_df和valid_y_df保存下来
        train_y_df.to_hdf(self.config.processed_train_data_path+"train_y.h5",key='train_y')
        valid_y_df.to_hdf(self.config.processed_train_data_path+"valid_y.h5",key='valid_y')
        print("train y df data saved!")
        return train_x,valid_x,train_y,valid_y
    
    def get_test_data(self):
        #获取测试数据
        self.normalized_data = self.normalized_data.sort_index(level=['date','code'])
        self.label = self.label.sort_index(level=['date','code'])

        test_x_df = self.normalized_data.loc[self.config.test_start_date:]
        test_y_df = self.label.loc[self.config.test_start_date:]

        if config.feature_nomalization == 'uniform':
            self.scaler = QuantileTransformer(output_distribution='uniform')
        elif config.feature_nomalization == 'normal':
            self.scaler = QuantileTransformer(output_distribution='normal')
        elif config.feature_nomalization == 'minmax':
            self.scaler = MinMaxScaler()
        elif config.feature_nomalization == 'robust':
            test_x_df = self.transform_remove_outliers(test_x_df,test_x_df.columns)
            self.scaler = MinMaxScaler()

        test_x_scaled = self.scaler.fit_transform(test_x_df[test_x_df.columns])
        test_x_df = pd.DataFrame(test_x_scaled, index=test_x_df.index, columns=test_x_df.columns)

        #获取测试数据的numpy数据
        print("transpoing test data...")
        test_x = self.transpose_df2array(test_x_df)
        test_y = self.transpose_df2array(test_y_df.to_frame())

        #把test_y_df保存下来
        test_y_df.to_hdf(self.config.processed_test_data_path+"test_y.h5",key='test_y')
        print("test y df data saved!")
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


def calculate_spearman(group):
    return spearmanr(group['Y_hat'], group['Y'])[0]


def get_result_index(data_index):
    """
    input: data_index是一个三维数组，形状为(样本数，timestep,2),最后一维分别是date和code
    function:根据data_index，仿照custom_dataloader中的方法，得到每个截面最后一天的数据标签
    output:(n,batchsize,2)
    """
    dates = data_index[:,:,0]
    data_list = list(np.unique(dates))

    # 获取所有唯一的日期
    unique_dates = np.unique(dates)

    # 为每个日期创建一个字典来保存数据
    date_groups ={date: [] for date in unique_dates}

    # 对每个日期进行分组
    for i, date in enumerate(dates[:,0]):  # 假设每个时间步的日期都相同
        date_groups[date].append(i)
    
    result_index = []
    for date in data_list:
        sample_indices = date_groups[date] # 该日所有股票在self.index中的索引
        sample_index = data_index[sample_indices] #该日所有股票在原始df中的索引 形状为（section_size,timestep,2）
        if sample_index.shape[0] != 0:
            last_date = sample_index[:,-1,:] #该日最后一个时间步的index
            result_index.append(last_date)
    return result_index

def save_test_data(config,Y_hat,test_Y,output_dir):
    """
    input:
        Y_hat为predict函数的输出结果，即模型预测结果，是一个dictionary，key为date，value为numpy数组，形状为(当日的股票个数，1)
        test_Y为实际的Y,为seires
    output:
        以predict_date为文件名将每一天的Y_hat和test_Y存储到output_dir中，并且求出每天截面的ic，再在时序上求均值后输出
        Y_hat在no_stock_date的日期中没有值
    """
    if isinstance(test_Y,pd.Series):
        df = test_Y.to_frame(name='Y')
    else:
        df = test_Y
        df.columns=['Y']
    #得到Y_hat的索引
    timestep = config.windowsize
    dates = df.index.get_level_values(0).unique().values[timestep-1:]
    stocks = df.index.get_level_values(1).unique().values

    # 展平数组，移除最后一个维度
    Y_hat = np.array(Y_hat)
    
    Y_hat = Y_hat[timestep-1:,:,:] #去掉前timestep-1个时间步
    arr_flattened = Y_hat.squeeze(-1)

    # 检查维度，确保对应的日期和股票数量与数组匹配
    assert len(dates) == arr_flattened.shape[0], "日期数量不匹配"
    assert len(stocks) == arr_flattened.shape[1], "股票数量不匹配"

    # 填充 'y_hat' 列
    for i, date in enumerate(dates):
        for j, stock in enumerate(stocks):
            # 只有当(date, stock)在DataFrame的索引中时，才进行赋值
            if (date, stock) in df.index:
                df.loc[(date, stock), 'Y_hat'] = arr_flattened[i, j]
    
    df = df.dropna()

    grouped_df = df.groupby(level='date')
    
    for name,group in grouped_df:
        group = group.reset_index(drop=False)
        group.to_csv(output_dir+name+'.csv')
    print("data saved!")
    
    #计算截面ic
    daily_correlation = grouped_df.apply(lambda x:x['Y_hat'].corr(x['Y']))
    average_coor = daily_correlation.mean()
    #计算rank ic
    spearman_correlations = grouped_df.apply(calculate_spearman)
    rank_avg = spearman_correlations.mean()
    return (average_coor,rank_avg)


#%%
config = Config()
def load_train_data(config=config,path = config.processed_train_data_path):
    train_x = np.load(os.path.join(path,"train_x.npy")) #(feature,date,code)
    valid_x = np.load(os.path.join(path,"valid_x.npy"))
    train_y = np.load(os.path.join(path,"train_y.npy"))
    valid_y = np.load(os.path.join(path,"valid_y.npy"))

    #从中选出需要用到的feature
    train_x = train_x[config.selected_feature_set,:,:]
    valid_x = valid_x[config.selected_feature_set,:,:]

    #print(train_y.head())
    #print(valid_y.head())
    train_and_valid_data = (train_x,valid_x,train_y,valid_y)

    return train_and_valid_data

def load_test_data(config=config,path=config.processed_test_data_path):
    test_x = np.load(os.path.join(path,"test_x.npy"))
    test_y = np.load(os.path.join(path,"test_y.npy"))

    test_x = test_x[config.selected_feature_set,:,:]

    return test_x,test_y




def cal_alpha(XY):
    #查看pnl如何计算
    d1 = XY.copy()
    enterRatio = 0.9
    exitRatio = 0.9
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

def get_XY(yest,xy):
    #xy是原始xy文件，yest是合并后的预测文件
    universe = 'univ_tradable'

    XY = xy.loc[xy[universe]==1,:'ud_limit_h4']
    XY= XY.rename(columns={'y1':'y'})
    XY = pd.merge(XY, yest,on=['date','code'],how='inner')

    ##---- 1. benchmark ----##
    XY['yest'] = XY['Y_hat']
    return XY


if config.save_processed_train_data == True:
    start_time = time.time()
    data_module = Data(config)
    end_time = time.time()
    print("time:",end_time-start_time)
    start_time = time.time()
    train_x,valid_x,train_y,valid_y = data_module.get_train_and_valid_data()

    end_time = time.time()
    print("time:",end_time-start_time)

    np.save(config.processed_train_data_path+"train_x.npy",train_x,allow_pickle=True)
    np.save(config.processed_train_data_path+"valid_x.npy",valid_x,allow_pickle=True)
    np.save(config.processed_train_data_path+"train_y.npy",train_y, allow_pickle=True)
    np.save(config.processed_train_data_path+"valid_y.npy",valid_y, allow_pickle=True)
    print("train data saved!")
       

    test_x,test_y = data_module.get_test_data()
    print(test_x.shape,test_y.shape)
    #print(test_x.head())
    #print(test_y.head())
    np.save(config.processed_test_data_path+"test_x.npy",test_x, allow_pickle=True)
    np.save(config.processed_test_data_path+"test_y.npy",test_y, allow_pickle=True)
    print("test data saved!")

else: 
    print("Loading train data...")
    train_and_valid_data = load_train_data(config,config.processed_train_data_path)

    print("Loading test data...")
    test_x,test_y = load_test_data(config,config.processed_test_data_path)

    print("Loading back test data...")
    #xy = pd.read_hdf("/home/laiminzhi/wenbin/DL_stock_combo/data/xy_data/xy_data_new.h5")
    #xy.index = xy.index.set_names(['code1','date','code'], level=[0, 1,2])
    # 删除不需要的索引层
    #xy = xy.droplevel('code1')
    print("Data loaded!")
    
np.random.seed(config.random_seed)

if config.do_train:
    print(f"Training...")
    config.input_size = train_and_valid_data[0].shape[0]

    train(config,train_and_valid_data)
    print("Train finished!")


if config.do_validation:
    train_x,valid_x,train_y,valid_y = train_and_valid_data
    print(f"Use validation data to predict...")
    y_hat= predict(config,valid_x)
    #print(len(y_hat),y_hat[0].shape)
    #加载valid_y的df文件
    valid_y_df = pd.read_hdf(config.processed_train_data_path+"valid_y.h5")
    ic = save_test_data(config,y_hat,valid_y_df,config.valid_save_path)
    print(f"ic={ic}")
    #calculate alpha
    all_files = [pd.read_csv(f'{config.valid_save_path}{f}',
                            dtype={'date':str})for f in sorted(os.listdir(config.valid_save_path))]
    yest = pd.concat(all_files, axis=0) #贴合所有的预测值
    XY = get_XY(yest,xy)
    alpha = cal_alpha(XY)
    print(f"alpha={alpha}")

if config.do_predict:
    print(f"Testing...")
    Y_hat_test = predict(config,test_x)
    #Y_hat = predict(config,train_and_valid_data[0]) #用训练集来预测
    #加载test_y的df文件
    test_y_df = pd.read_hdf(config.processed_test_data_path+"test_y.h5")
    ic = save_test_data(config,Y_hat_test,test_y_df,config.predict_save_path)
    print(f"ic={ic[0]},rank ic = {ic[1]}")
    #calculate alpha
    all_files = [pd.read_csv(f'{config.predict_save_path}{f}',
                            dtype={'date':str})for f in sorted(os.listdir(config.predict_save_path))]
    yest = pd.concat(all_files, axis=0) #贴合所有的预测值
    XY = get_XY(yest,xy)
    alpha = cal_alpha(XY)
    print(f"alpha={alpha}")
