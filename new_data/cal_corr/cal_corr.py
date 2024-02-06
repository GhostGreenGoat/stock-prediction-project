#%%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
#%%
df = pd.read_hdf('/home/laiminzhi/data/xydata/xy_data_3355.h5')
selected_feature = df.columns[df.columns.str.contains('x')].to_list()
selected_y = ['y1_label','y3_label']
ud_limit = ['ud_limit_h2','univ_tradable']
selected_feature = selected_y+ud_limit+selected_feature
df = df.loc[:,selected_feature]
df = df.replace([-np.inf,np.inf],0.0)
#筛选出universe中的数据
df = df[df['univ_tradable'] == 1]
#去掉涨跌停数据

df = df[df['ud_limit_h2'] == 0]
# %%
#计算df中x开头的格列和y1_label,y3_label的相关系数
df = df[df.columns[df.columns.str.contains('x')|df.columns.str.contains('y1_label')|df.columns.str.contains('y3_label')]]
#%%
#先用subdf测试
pearson_corr = df.corr(method='pearson')
spearman_corr = df.corr(method='spearman')
pearson_corr.to_csv('./xy_data_3355_pearson.csv')
spearman_corr.to_csv('./xy_data_3355_spearman.csv')

# %%
def draw_heatmap(correlation_matrix,name=None):
    # 设置图的大小
    plt.figure(figsize=(20, 20))

    # 绘制热图
    sns.heatmap(correlation_matrix, 
                annot=False,        # 如果你希望在每个格子内显示相关系数，则将此设置为True
                cmap='coolwarm',    # 选择一个颜色映射方案
                center=0,           # 设置色彩条中心点的值
                square=True,        # 如果你希望每个格子都是正方形，则将此设置为True
                linewidths=.5,      # 设置格子之间线的宽度
                cbar_kws={"shrink": .5})  # 设置色彩条的参数

    # 设置图的其他参数，如标题等
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()  # 调整整体布局

    # 显示图像
    plt.show()
    plt.savefig(name,format='pdf')
#%%
draw_heatmap(pearson_corr,'./xy_data_3355_pearson.pdf')
draw_heatmap(spearman_corr,'./xy_data_3355_spearman.pdf')
#%%
def draw_bar(correlation_vector):
    #画各x值和y的相关系数柱状图
    plt.figure(figsize=(20, 20))
    plt.bar(x=correlation_vector.index,height=correlation_vector.values)
    plt.show()
    plt.savefig('bar.pdf',format='pdf')
#%%
draw_bar(pearson_corr.loc['y1_label'][2:],'./xy_data_3355_y1_label_pearson.pdf')
draw_bar(pearson_corr['y3_label'][2:],'./xy_data_3355_y3_label_pearson.pdf')
draw_bar(spearman_corr.loc['y1_label'][2:],'./xy_data_3355_y1_label_spearman.pdf')
draw_bar(spearman_corr['y3_label'][2:],'./xy_data_3355_y3_label_spearman.pdf')
# %%
#查看计算结果
pearson_result = pd.read_csv('./xy_data_3355_pearson.csv',index_col=0)
spearman_result = pd.read_csv('./xy_data_3355_spearman.csv',index_col=0)
# %%
#查看我筛选的特征的结果
selected_x = pd.read_csv('../selected_feature.csv',index_col=0)['AlphaName'].to_list()+['y1_label','y3_label']
#selected_x = [x for x in pearson_result.columns if x in selected_x]
pearson_result_selected = pearson_result.loc[selected_x,selected_x]
spearman_result_selected = spearman_result.loc[selected_x,selected_x]
# %%
draw_heatmap(pearson_result_selected,'71feature_pearson.pdf')
draw_heatmap(spearman_result_selected,'71feature_spearman.pdf')
# %%
draw_bar(pearson_result_selected.loc['y1_label'][:-2].sort_values())
draw_bar(pearson_result['y1_label'][2:].sort_values())
draw_bar(spearman_result_selected.loc['y1_label'][:-2].sort_values())
draw_bar(spearman_result['y1_label'][2:].sort_values())
# %%
#查看特征相关性的分布
from scipy.stats import gaussian_kde

def draw_pdf(vector):

    #vector1 = (vector1 - vector1.min()) / (vector1.max() - vector1.min())
    #vector2 = (vector2 - vector2.min()) / (vector2.max() - vector2.min())


    # 计算每个向量的PDF
    kde1 = gaussian_kde(vector)


    # 设置绘图的点
    x_min = vector.min()
    x_max = vector.max()
    x = np.linspace(x_min, x_max, 1000)

    # 绘制两个向量的PDF
    plt.figure(figsize=(8, 6))
    plt.plot(x, kde1(x), label='pred PDF')

    # 添加图例
    plt.legend()

    # 显示图
    plt.show()

draw_pdf(pearson_result_selected.loc['y1_label'][:-2])
draw_pdf(pearson_result['y1_label'][2:])

draw_pdf(spearman_result_selected.loc['y1_label'][:-2])
draw_pdf(spearman_result['y1_label'][2:])
# %%
