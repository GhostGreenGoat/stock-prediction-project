#%%
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pdb
import pickle
import gc
import scipy.stats as st

def trim(x, rlow=None, rhigh=None,ratio=None,drop_nan=True): # 0<ratio<0.5
    x1 = x.copy()
    if ratio != None:
        if drop_nan:
            x2 = x1.loc[x1.notna()]
            rlow = np.percentile(x2.values,ratio*100)
            rhigh = np.percentile(x2.values, (1-ratio)*100)
        else:
            print('[ERROR] nan in x values!')
            raise BaseException
    x1[x1<rlow] = rlow
    x1[x1>rhigh] = rhigh
    return x1

def cal_norm(ds):
    ds1 = ds.rank(pct=False,ascending=True,na_option='keep')/(ds.notna().sum()+1)
    ds2 = pd.Series(st.norm().ppf(ds1),index=ds.index)
    return ds2

def calTurnover(flagMat,isSummary=True):
## flag.mat contains 1, -1, NA;
## turnover=1 means we trade 1 (buy 0.5 and sell 0.5) every day;
    flagMat = flagMat.fillna(0)
    posiNum = (flagMat!=0).sum(axis=1)
    
    tradeMat = (flagMat - flagMat.shift(1,axis=0)).abs()
    to = (tradeMat.sum(axis=1)/(posiNum+posiNum.shift(1))*2).fillna(0)
    if isSummary:
        to = '%.3g'%to.mean()
    return to

def draw_pnl(rtn_vec,fig_name = 'backtest'):
    alpha = rtn_vec.mean()*1e4
    sigma = rtn_vec.std()*1e4
    sharpe = alpha/sigma *np.sqrt(252)
    win_rate = (rtn_vec>0).sum()/(rtn_vec.notna().sum())

    dd = (rtn_vec.cumsum() - rtn_vec.cumsum().cummax())
    mdd = dd.min()*1e2
    mdd_date = dd.idxmin()
    
    fig_name = '{0}:alpha={1};sigma={2};sharpe={3};win_rate={4};mdd={5}% on {6}'.format(
            fig_name,'%.2f'%alpha,'%.2f'%sigma,'%.2f'%sharpe,'%.2f'%win_rate,
            '%.2f'%mdd,mdd_date)

    rtn_vec.cumsum().plot(title=fig_name)
    res = pd.Series([alpha,sigma,sharpe,win_rate,mdd,mdd_date],
                        index=['alpha','sigma','sharpe','win_rate','mdd','mdd_date'])
    return res

def plotPNL(rtnMat,posiMat,posiMat_nature='position',costbps=10,figName='backtest',savefigFile=None):
    '''
    posiMat:contains[0,1],no np.nan;
    '''
    ## 1) calculate pnlVec, costVec;
    pnlMat = rtnMat * posiMat
    if posiMat_nature=='position':
        pnlVec = pnlMat.sum(axis=1)/(posiMat==1).sum(axis=1)
        turnoverVec = calTurnover(posiMat,isSummary=False)
    elif posiMat_nature=='weight':
        pnlVec = pnlMat.sum(axis=1)
        turnoverVec = abs( posiMat.fillna(0) - posiMat.fillna(0).shift(1,axis=0)).sum(axis=1)
    else:
        print('[ERROR] wrong posiMat_nature value!')
        return False

    costVec = turnoverVec * costbps * 1e-4
    pnl_afterCostVec = pnlVec - costVec
    
    ## 2) calculate indicators & pnl(DataFrame);
    alpha = pnlVec.mean()*1e4
    alpha2 = pnl_afterCostVec.mean()*1e4
    sigma = pnlVec.std()*1e4
    sharpe = alpha/sigma*np.sqrt(252)
    sharpe2 = pnl_afterCostVec.mean()/pnl_afterCostVec.std()*np.sqrt(252)
    stkNum = posiMat.fillna(0).apply(lambda x: (x!=0).sum(),axis=1).mean()
    turnover = turnoverVec.mean()
    win_rate = (pnl_afterCostVec>0).sum()/(pnl_afterCostVec.notna().sum())

    pnl = pd.DataFrame(pnlVec,columns=['pnl'])
    pnl['cum_pnl'] = pnlVec.cumsum()
    pnl['cost'] = costVec
    pnl['cum_cost'] = costVec.cumsum()
    pnl['pnl_afterCost'] = pnl_afterCostVec
    pnl['cum_pnl_afterCost'] = pnl_afterCostVec.cumsum()
    pnl['drawdown'] = pnl['cum_pnl'] - pnl['cum_pnl'].cummax()
    pnl.reset_index(inplace=True)
    pnl['date'] = pd.to_datetime(pnl['date'])
    pnl.set_index('date',inplace=True)
    
    mdd = pnl['drawdown'].min()*1e2
    mdd_date = pnl['drawdown'].idxmin()

    ## 3) plot PNL: pd.Series.plot() + pd.Series.plot.bar()
    fig = plt.figure(figsize=(10, 6))
    ax1 = fig.add_axes([0,0.4,1,1]) ## 第1个子图；
    pnl['cum_pnl'].plot(color='blue') ##pd.Series.plot()能够自动识别日期类index，并作为xlable;（并且会自动选择合适的采样）
    pnl['cum_pnl_afterCost'].plot(color='red')
    pnl['cum_cost'].plot(color='green')
    ax2 = fig.add_axes([0,0,1,0.25]) ## 第2个子图；
    pnl['drawdown'].plot.bar(color='gray') ##绘制plot.bar时，xlable并不会自动选择合适的采样；

    ax1.grid(True,axis='x',color='green',linestyle=':',linewidth=0.5) ## 绘制辅助线；
    ax2.grid(True,axis='x',color='green',linestyle=':',linewidth=0.5)

    ax2.xaxis.set_visible(False) ##xlable太密集，不显示
    ax1.set_xlabel('')
    thisName = '{0}:alpha={1};sigma={2};sharpe={3};stknum={4};turnover={5};win_rate={6};mdd={7}% on {8}'.format(
        figName,'%.2f'%alpha,'%.2f'%sigma,'%.2f'%sharpe,'%.2f'%stkNum,'%.2f'%turnover,'%.2f'%win_rate,
        '%.2f'%mdd,mdd_date.strftime('%Y%m%d'))
    ax1.set_title(thisName)
    ax1.legend(loc='upper left',labels=['pnl','pnl_afterCost','cost'])
    if savefigFile != None:
        fig.savefig(savefigFile,dpi=100,bbox_inches='tight')
    res = pd.Series([alpha,sigma,sharpe,alpha2,sharpe2,stkNum,turnover,win_rate,mdd,mdd_date.strftime('%Y%m%d')],
                    index=['alpha','sigma','sharpe','alpha2','sharpe2','stkNum','turnover','win_rate','mdd','mdd_date'])
#     plt.clf()
#     plt.close(fig)
    return {'pnl':pnl,'pnl_summary':res}

def cal_topPNL(yestData,enterRatio=0.8,exitRatio=0.8,year='all',costbps=10,savefigFile=None):    
    if (not all([x in yestData.columns for x in ['date','code','yest','y']])):
        print("[ERROR] yestData's format is wrong!")
        return False
    d1 = yestData.loc[:,['date','code','yest','y']]
    
    #pdb.set_trace()
    ## 1) calculate yestRank & posi;
    d1['yestRank'] = d1.groupby('date')['yest'].rank(method='average',na_option='keep',ascending=True,pct=True)
    
    ## 2) rtnMat & posiMat;
    rtnMat = pd.pivot_table(data=d1,index='date',columns='code',values='y',dropna=False)
    yestRankMat = pd.pivot_table(data=d1,index='date',columns='code',values='yestRank',dropna=False)
    posiMat = pd.DataFrame(np.full(yestRankMat.shape,fill_value=np.nan),index=yestRankMat.index,columns=yestRankMat.columns)
    posiMat[yestRankMat>enterRatio] = 1
    posiMat[yestRankMat<exitRatio] = 0
    posiMat = posiMat.fillna(method='ffill',axis=0).fillna(0)
    posiMat[yestRankMat.isna()] = np.nan
    
    ## 3) result
    if (year=='all'):
        res = plotPNL(rtnMat,posiMat,costbps=costbps,figName=year,savefigFile=savefigFile)
        res['pnl_summary'] = res['pnl_summary'].to_frame(name='all').T
    elif(year=='each'):
        all_year = rtnMat.index.str.slice(0,4).unique()
        pnl_list = []
        pnl_sum_dict = {}
        for tmp_year in all_year:
            tmp_posiMat = posiMat.loc[posiMat.index.str.slice(0,4)==tmp_year,:]
            tmp_rtnMat = rtnMat.loc[rtnMat.index.str.slice(0,4)==tmp_year,:]
            tmp = plotPNL(tmp_rtnMat,tmp_posiMat,costbps=costbps,figName=tmp_year,savefigFile=savefigFile)
            pnl_list.append(tmp['pnl'])
            pnl_sum_dict[tmp_year] = tmp['pnl_summary']
        res = {}
        res['pnl'] = pd.concat(pnl_list,axis=0)
        res['pnl_summary'] = pd.DataFrame(pnl_sum_dict).T
    else:
        tmp_posiMat = posiMat.loc[posiMat.index.str.slice(0,4)==year,:]
        tmp_rtnMat = rtnMat.loc[rtnMat.index.str.slice(0,4)==year,:]
        res = plotPNL(tmp_rtnMat,tmp_posiMat,costbps=costbps,figName=year,savefigFile=savefigFile)
        res['pnl_summary'] = res['pnl_summary'].to_frame(name='all').T
    return(res)

def cal_topPNL2(yestData,ud_field='ud_limit',enterRatio=0.8,exitRatio=0.8,year='all',costbps=10,savefigFile=None):
    ## no buy if up_limit && no sell if down_limit;
    if (not all([x in yestData.columns for x in ['date','code','yest','y']])):
        print("[ERROR] yestData's format is wrong!")
        return False
    d1 = yestData.copy()

    ## 1) calculate yestRank;
    d1['yestRank'] = d1.groupby('date')['yest'].rank(method='average',na_option='keep',ascending=True,pct=True)
    rtnMat = pd.pivot_table(data=d1,index='date',columns='code',values='y',dropna=False)
    yestMat = pd.pivot_table(data=d1,index='date',columns='code',values='yest',dropna=False)
    yestRankMat = pd.pivot_table(data=d1,index='date',columns='code',values='yestRank',dropna=False).fillna(0)
    posiMat = pd.DataFrame(np.full(yestRankMat.shape,fill_value=0),index=yestRankMat.index,columns=yestRankMat.columns)
    ud_limitMat = pd.pivot_table(data=d1,index='date',columns='code',values=ud_field,dropna=False).fillna(0)

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
        
    #pdb.set_trace()
    ## 3) result
    if (year=='all'):
        res = plotPNL(rtnMat,posiMat,costbps=costbps,figName=year,savefigFile=savefigFile)
        res['pnl_summary'] = res['pnl_summary'].to_frame(name='all').T
    elif(year=='each'):
        all_year = rtnMat.index.str.slice(0,4).unique()
        pnl_list = []
        pnl_sum_dict = {}
        for tmp_year in all_year:
            tmp_posiMat = posiMat.loc[posiMat.index.str.slice(0,4)==tmp_year,:]
            tmp_rtnMat = rtnMat.loc[rtnMat.index.str.slice(0,4)==tmp_year,:]
            tmp = plotPNL(tmp_rtnMat,tmp_posiMat,costbps=costbps,figName=tmp_year,savefigFile=savefigFile)
            pnl_list.append(tmp['pnl'])
            pnl_sum_dict[tmp_year] = tmp['pnl_summary']
        res = {}
        res['pnl'] = pd.concat(pnl_list,axis=0)
        res['pnl_summary'] = pd.DataFrame(pnl_sum_dict).T
    else:
        tmp_posiMat = posiMat.loc[posiMat.index.str.slice(0,4)==year,:]
        tmp_rtnMat = rtnMat.loc[rtnMat.index.str.slice(0,4)==year,:]
        res = plotPNL(tmp_rtnMat,tmp_posiMat,costbps=costbps,figName=year,savefigFile=savefigFile)
        res['pnl_summary'] = res['pnl_summary'].to_frame(name='all').T
    return(res)

def cal_posPNL(yestData,y_type='y5'):## calculate positive weight pnl;
    def _cal_posPNL(df):
        df['yest_norm'] = cal_norm(df['yest'])
        df['flag'] = (df['yest_norm']>0) & (df['ud_limit_h2']!=1)
        res = (df[y_type] * df['yest_norm'] * df['flag']).sum()/(df['yest_norm'] * df['flag']).sum()
        return res
    
    pnl = yestData.groupby('date').apply(_cal_posPNL)
    alpha = pnl.mean()*1e4
    sigma = pnl.std()*1e4
    sharpe = alpha/sigma*np.sqrt(252)
    pnl_summary = pd.Series([alpha,sigma,sharpe,],index=['alpha','sigma','sharpe',])
    return {'pnl':pnl,'pnl_summary':pnl_summary}

def cal_scorecard(df,y_type = 'y5',method = 'pearson'):
    ic = df.groupby('date')[['yest',y_type]].apply(lambda x:x.corr(method=method).iloc[0,1])
    ic_mean = ic.mean()
    ic_std = ic.std()
    icir = ic_mean/ic_std*pow(250,0.5)
    return {'ic':ic,'ic_mean':ic_mean,'icir':icir}

def cal_multiPNL(yestData):
    top_pnl = cal_topPNL(yestData,enterRatio=0.95,exitRatio=0.85,year='all',costbps=10,savefigFile=None)
    top_pnl2 = cal_topPNL2(yestData,ud_field='ud_limit_h2',enterRatio=0.95,exitRatio=0.85,year='all',costbps=10,savefigFile=None)  #
    pos_pnl = cal_posPNL(yestData,y_type='y5')
    ic_test = yestData[['date','y5','yest']].set_index('date')
    res_indicator = cal_scorecard(ic_test,y_type = 'y5',method = 'pearson')
    res_all = {'res_indicator':res_indicator,'top_pnl':top_pnl,'top_pnl2':top_pnl2,'pos_pnl':pos_pnl}
    return res_all

def simple_ols(yestData, yest1='yest_benchmark', yest2='yest_pool', weight1=0.4):
    yest_norm = yestData.groupby('date')[[yest1, yest2]].apply(lambda x: (x - x.mean()) / x.std())
    weight2 = 1 - weight1
    yest = weight1 * yest_norm[yest1] + weight2 * yest_norm[yest2]
    return yest
    
def cal_margin(m_old,m_new):
    s01 = m_old['top_pnl2']['pnl_summary'].loc['all',['alpha','sharpe2']]
    s02 = m_old['pos_pnl']['pnl_summary'][['alpha','sharpe']]
    m0 = s01.append(s02)
    s11 = m_new['top_pnl2']['pnl_summary'].loc['all',['alpha','sharpe2']]
    s12 = m_new['pos_pnl']['pnl_summary'][['alpha','sharpe']]
    m1 = s11.append(s12)
    margin_top = (s11/s01).sum()/2
    margin_pos = (s12/s02).sum()/2
    margin = (m1/m0).sum()/4
    return pd.Series({'margin_top':margin_top,'margin_pos':margin_pos,'margin':margin})


#%%
all_files = [pd.read_csv(f'/home/laiminzhi/reconfiguration_code/predict_data/y1_label__all_minmax_no_ud_limit_h2/selected_feature(all)_loss(advance)_depth4_sizes(249_128_128_1)_bd/{f}', dtype={'date':str})for f in sorted(os.listdir('/home/laiminzhi/reconfiguration_code/predict_data/y1_label__all_minmax_no_ud_limit_h2/selected_feature(all)_loss(advance)_depth4_sizes(249_128_128_1)_bd'))]
yest = pd.concat(all_files, axis=0)
xy = pd.read_hdf('/home/laiminzhi/wenbin/DL_stock_combo/data/xy_data/xy_data.h5').reset_index()
universe = 'univ_tradable'

xy = xy.loc[xy[universe]==1,:'ud_limit_h4']
xy = xy.rename(columns={'y1':'y'})
xy = pd.merge(xy, yest,on=['date','code'],how='inner')

##---- 1. benchmark ----##
xy['yest'] = xy['Y_hat']
m_pool = cal_multiPNL(xy)
#xy['yest'] = xy['Y']
#m_baseline = cal_multiPNL(xy)
#xy['yest_old'] = simple_ols(xy, yest1='yest_baseline', yest2='yest_pool', weight1=0.4)
#xy['yest'] = xy['yest_old']
#m_old = cal_multiPNL(xy)
#%%
##---- 2. yest_new ----##
all_files_new = [pd.read_csv(f'/home/laiminzhi/wenbin/DL_stock_combo/data/prediction_data_new/{f}', dtype={'date':str}) for f in sorted(os.listdir('/home/laiminzhi/wenbin/DL_stock_combo/data/prediction_data_new'))]
yest_new = pd.concat(all_files_new, axis=0)
xy = pd.merge(xy, yest_new, on=['date','code'], how='inner')
xy['yest'] = xy['yest_new']
m_new = cal_multiPNL(xy)
#%%
##---- 3. add yest_new ----##
xy['yest_after'] = simple_ols(xy, yest1='yest_new', yest2='yest_old', weight1=0.4)
xy['yest'] = xy['yest_after']
m_after = cal_multiPNL(xy)
#%%
##---- 4. whether improve ----##
margin_baseline = cal_margin(m_baseline, m_new)
print('margin_baseline:\n', margin_baseline)

margin = cal_margin(m_old, m_after)
print('margin:\n', margin)
# %%
print(m_new['res_indicator'])
# %%
