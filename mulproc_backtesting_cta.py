#coding:utf-8
'''
Walk Forward Backtesting for CTA Trending Strategy
'''
import os
import os.path
import re
import random
import itertools
import numpy as np
import pandas as pd
from pandas import DataFrame
from datetime import datetime
from multiprocessing import Pool, Manager
from concurrent.futures import ProcessPoolExecutor


TIME_LIST = []
DATE_LIST = []
PRODUCT_LIST = []
INDEX_DICT = {}
PRINCIPAL = 10000
SCREENED_SIZE = 15
PORTOFOLIO_SIZE = 15
OPTIMIZE_PORT = False

#初始化

def init_datesANDtimesANDproducts(path='./'):
    global TIME_LIST, DATE_LIST, PRODUCT_LIST
    TIME_LIST = []
    DATE_LIST = []
    PRODUCT_LIST = []
    #得到规范化交易时间点列表
    with open('%strading_time.txt' % path) as file:
       	for time in file.readlines():
      		TIME_LIST.append(re.findall('[0-9]+',time)[0])
    #得到有记录的交易日列表
    for _p, dirnames, _f in os.walk('%sindex/' % path):
       	for dirname in  dirnames:
      		DATE_LIST.append(dirname)
    #得到可交易的产品列表
    for _p, _d, filenames in os.walk('%sindex/20150407' % path):
       	for filename in filenames:
      		PRODUCT_LIST.extend(re.findall('^[a-zA-Z]+', filename))

#交易回测函数

def fast_day_processing(df, window, threshold, remaining=False, *lastday):
    '''
    df:DataFrame with the info of index_1, index_2, Signal, Moving_Average
    remaining is True: Based on the remaining position to trade
    otherwise: the first day should not consider the threshold
    '''
    #df['Signal_MA'] = df['Signal'].rolling(window=window).mean()#400us
    df['Above_MA'] = 0
    df['Action'] = 0
    df['Position_1'] = np.nan
    df['Position_2'] = np.nan
    df['Daily_Profit'] = np.nan

    arr = df.to_records()
    rows = len(arr)
    flag = 0

    if remaining:
        arr[0]['Position_1'] = lastday[2]
        arr[0]['Position_2'] = lastday[3]
        arr[0]['Daily_Profit'] = arr[0]['Position_1'] * (arr[0][1] - lastday[0]) + \
                                    arr[0]['Position_2'] * (arr[0][2] - lastday[1])
    else:
        arr[0]['Above_MA'] = 1 if arr[0]['Signal_MA'] < arr[0]['Signal'] else -1

    for r in range(1, rows, 1):
        #判断Signal与MA的关系
        if arr[r]['Signal'] < (1 - threshold) * arr[r]['Signal_MA']:
            arr[r]['Above_MA'] = -1
        elif (1 + threshold) * arr[r]['Signal_MA'] < arr[r]['Signal']:
            arr[r]['Above_MA'] = 1
        #用Above_MA来决定action,注意action是延后一个时间单位
        if arr[r - 1]['Above_MA']  != 0 and arr[r - 1]['Above_MA'] != flag:
            arr[r]['Action'] = arr[r - 1]['Above_MA']
            flag = arr[r]['Action']
        #设置仓位
        if arr[r]['Action'] != 0:
            arr[r]['Position_1'] = arr[r]['Action'] * PRINCIPAL / arr[r - 1][1]
            arr[r]['Position_2'] = - arr[r]['Action'] * PRINCIPAL / arr[r - 1][2]
        else:
            arr[r]['Position_1'] = arr[r - 1]['Position_1']
            arr[r]['Position_2'] = arr[r - 1]['Position_2']
        #计算P&L
        arr[r]['Daily_Profit'] = arr[r]['Position_1'] * (arr[r][1] - arr[r - 1][1]) + \
                                    arr[r]['Position_2'] * (arr[r][2] - arr[r - 1][2])

    df = DataFrame.from_records(arr, index='index')
    df['Cumul_Profit'] = df['Daily_Profit'].cumsum()
    return df

def fast_minute_processing(day_data, minute_data, window, threshold, remaining=False, *lastday):
    '''
    day_data: 记录有window-1的t-1日MA，记录在t日
    minute_data: 交易开始至结束所有的分钟数据
    window, threshold: 交易参数
    remainning: True则为接着之前的仓位继续交易
                False则重新开仓，第一个分钟不考虑阈值，直接根据ma线进入
    '''
    def __compute_daily__(ma_sr, min_df):
        date_ = ma_sr.name.strftime('%Y%m%d')
        return min_df.loc[date_, 'Profit'].sum()
    #day_data['MA_t-1'] = day_data['Signal'].rolling(window).apply(lambda x: (x[:-1].sum()) / (window - 1))#1.5ms
    minute_data['MA_Minute'] = 0.0
    minute_data['Above_MA'] = 0
    minute_data['Action'] = 0
    minute_data['Position_1'] = np.nan
    minute_data['Position_2'] = np.nan
    minute_data['Profit'] = np.nan

    arr = minute_data.to_records()
    date_ = datetime.date(arr[0][0])
    arr[0]['MA_Minute'] = (day_data.loc[date_,'MA_t-1'] * (window - 1) + arr[0]['Signal']) / window
    rows = len(arr)
    flag = 0
    if remaining:
        arr[0]['Position_1'] = lastday[2]
        arr[0]['Position_2'] = lastday[3]
        arr[0]['Profit'] = arr[0]['Position_1'] * (arr[0][1] - lastday[0]) + \
                            arr[0]['Position_2'] * (arr[0][2] - lastday[1])
    else:
        arr[0]['Above_MA'] = 1 if arr[0]['Signal'] > arr[0]['MA_Minute'] else -1


    for r in range(1, rows, 1):
        #得到分钟线的MA
        date_ = datetime.date(arr[r][0])
        arr[r]['MA_Minute'] = (day_data.loc[date_,'MA_t-1'] * (window - 1) + \
                                arr[r]['Signal']) / window
        #根据threshold判断信号
        if (1 + threshold) * arr[r]['MA_Minute'] < arr[r]['Signal']:
            arr[r]['Above_MA'] = 1
        elif arr[r]['Signal'] < (1 - threshold) * arr[r]['MA_Minute']:
            arr[r]['Above_MA'] = -1
        else:
            arr[r]['Above_MA'] = 0
        #根据信号得到Action
        if arr[r - 1]['Above_MA'] != 0 and arr[r - 1]['Above_MA'] != flag:
            arr[r]['Action'] = arr[r - 1]['Above_MA']
            flag = arr[r]['Action']
        #设置仓位
        if arr[r]['Action'] != 0:
            arr[r]['Position_1'] = arr[r]['Action'] * PRINCIPAL / arr[r - 1][1]
            arr[r]['Position_2'] = - arr[r]['Action'] * PRINCIPAL / arr[r - 1][2]
        else:
            arr[r]['Position_1'] = arr[r - 1]['Position_1']
            arr[r]['Position_2'] = arr[r - 1]['Position_2']
        #计算P&L
        arr[r]['Profit'] = arr[r]['Position_1'] * (arr[r][1] - arr[r - 1][1]) + \
                            arr[r]['Position_2'] * (arr[r][2] - arr[r - 1][2])

    minute_data = DataFrame.from_records(arr, index='index')
    #统计分钟交易每天的收益和累积收益
    day_data['Daily_Profit'] = day_data.apply(lambda x: __compute_daily__(x, minute_data), axis=1)
    day_data['Cumul_Profit'] = day_data['Daily_Profit'].cumsum()
    return (day_data, minute_data)

#若干辅助函数
def delta_mon(start, end):
    start_mon = int(start) / 100 % 100
    start_year = int(start) / 10000
    end_mon = int(end) / 100 % 100
    end_year = int(end) / 10000
    return (end_year - start_year) * 12 + end_mon - start_mon

def get_next_mon(start, mon):
    start_day = int(start) % 100
    start_mon = int(start) / 100 % 100
    start_year = int(start) / 10000
    year = start_year + (start_mon + mon - 1) / 12
    month = start_mon + mon - ((start_mon + mon - 1) / 12) * 12
    date = datetime(year, month, start_day)
    return datetime.strftime(date, '%Y%m%d')

def iter_all_products():
    '''
    path: denotes the file path contains the product name
    return the tuple containing all possible combinations of two products
    '''
    global PRODUCT_LIST
    product_list = list(PRODUCT_LIST)
    nums = len(product_list)
    pair_list = []
    for i in range(nums-1):
        for j in range(i + 1, nums, 1):
            pair_list.append((product_list[i], product_list[j]))
    return tuple(pair_list)

def get_date_by_str(string):

    return datetime.date(datetime.strptime(string, '%Y%m%d'))

def split_period(start):
	'''
	Caution: the start day should less than 28!
	return the seven month timestamps
	'''
	start_year = int(start) / 10000
	start_month = int(start) / 100 % 100
	day = int(start) % 100
	periods = [start]
	for idx in range(1, 7, 1):
		mon = start_month + idx
		year = start_year
		if mon > 12:
			year += 1
			mon -= 12
		periods.append(datetime(year, mon, day).strftime('%Y%m%d'))
	return tuple(periods)

def get_windows(good_ma):
    if len(good_ma) == 1:
        return np.linspace(good_ma[0]-4, good_ma[0]+4, num=5)
    elif len(good_ma) == 2:
        return np.append(np.linspace(good_ma[0]-2, good_ma[0]+2, num = 3), \
                            np.linspace(good_ma[1]-2, good_ma[1]+2, num = 3))
    elif len(good_ma) == 3:
        a = np.append(np.linspace(good_ma[0]-4, good_ma[0]+4, num = 3), \
                        np.linspace(good_ma[1]-4, good_ma[1]+4, num=3))
        return np.append(a, np.linspace(good_ma[2]-4, good_ma[2]+4, num = 3))
    else:
        raise NameError('good_ma exceeded!')

def next_performance(mean_train, sharpe_train, mean_y, vol_y):
	if mean_train > mean_y:
	    return 0
	if sharpe_train > mean_y / vol_y * (250 ** 0.5):
	    return 0
	return 1

def get_start_by_window(start, window, mp_date_list=None):
    '''
    start: '20170707'
    '''
    if mp_date_list:
        date_list = map(int, list(mp_date_list))
    else:
        global DATE_LIST
        date_list = map(int, list(DATE_LIST))
    start = int(start)

    flag = False
    for idx, item in enumerate(date_list):
        if item == start:
            if idx - window + 1 >= 0:
                return str(date_list[idx - window + 1])
            else:
                return str(date_list[0])
        elif item < start and not flag:
            flag = True
        elif start < item and flag:
            if idx - window + 1 >= 0:
                return str(date_list[idx - window + 1])
            else:
                return str(date_list[0])
        elif start < item and not flag:
            return str(date_list[0])
    return str(date_list[-window])

def get_good_ma(start, end, best_pairs, mp_index_dict = None, mp_date_list=None, path='./'):
    '''
    用日收盘跑， 得到各产品对最优的三组参数中的MA

    '''
    Pair_paras = {}
    start_date = get_date_by_str(start)
    basic_windows = [10, 20, 30, 40, 50]
    basic_thresholds = [0, 0.002, 0.004, 0.006, 0.008]
    if mp_date_list:
        adj_start = get_start_by_window(start, 60, mp_date_list)
    else:
        adj_start = get_start_by_window(start, 60)

    for pair_str in best_pairs:
        pair = tuple(re.findall('[a-zA-Z]+', pair_str))
        basic_info = DataFrame(index=[], columns=['MA', 'Threshold', 'Sharpe_Ratio', 'P&L'])

        if mp_index_dict:
			_day_data = gen_pair_records(adj_start, end, pair, 'day', mp_index_dict)
        else:
			_day_data = gen_pair_records(adj_start, end, pair, 'day')

        for w in basic_windows:
            __day_data = _day_data.copy(deep=True)
            __day_data['Signal_MA'] = __day_data['Signal'].rolling(window=w).mean()
            assert not __day_data.loc[start_date:, 'Signal_MA'].isnull().any(), 'Moving Average Error!'
            for t in basic_thresholds:
                traded = fast_day_processing(__day_data.loc[start_date:].copy(deep=True), w, t)
                sharpe = traded['Daily_Profit'].mean() / traded['Daily_Profit'].std() * (250 ** 0.5)
                profit = traded['Cumul_Profit'].iloc[-1]
                info = {'MA': w, 'Threshold': t, 'Sharpe_Ratio': sharpe, 'P&L': profit}
                basic_info = basic_info.append(info, ignore_index=True)
        good_para = basic_info.sort_values(by='Sharpe_Ratio', ascending=False).iloc[:3]['MA']
        ma_ = set()
        for ma in list(good_para):
            ma_.add(int(ma))
        Pair_paras[pair_str] = ma_
    return Pair_paras

#生成价格数据与获取价格数据

def gen_index_info(path='./'):
    '''
    将所有的产品数据读入至全局变量INDEX_DICT中，方便后续回测时调用
    如果没有整合数据，则在目录下迭代整合各个产品的分钟和日收盘数据，保存至./merged_index/
    如果已经整合过，直接读取该目录下的.csv文件
    '''
    global INDEX_DICT
    INDEX_DICT = {}
    start = DATE_LIST[0]
    end = DATE_LIST[-1]
    time_list = list(TIME_LIST)
    dates = [get_date_by_str(d) for d in DATE_LIST]
    if 'merged_index' in os.listdir(path):
        for index in PRODUCT_LIST:
            Day_data = pd.read_csv('%smerged_index/%s/day.csv' % (path, index), header=0, index_col=0, parse_dates=True)
            Minute_data = pd.read_csv('%smerged_index/%s/minute.csv' % (path, index), header=0, index_col=0, parse_dates=True)
            INDEX_DICT[index] = (Day_data.copy(deep=True), Minute_data.copy(deep=True))

    else:
        os.mkdir('%smerged_index' % path)
        for index in PRODUCT_LIST:
            Day_data = DataFrame(columns=['%s' % index], index=dates)
            Minute_data = DataFrame(columns=['%s' % index])
            for date in DATE_LIST:
                _dir = path + 'index/' + date + '/' + index + '.csv'
                _datetime_dict = {int(t): datetime.strptime(date + t, '%Y%m%d%H%M') for t in time_list}
                _datetime_list = [datetime.strptime(date + t, '%Y%m%d%H%M') for t in time_list]
                try:
                    _df = pd.read_csv(_dir, header=None)
                    _df.drop([2], axis=1, inplace=True)
                    _df.rename(columns={0: 'Time', 1: index}, inplace=True)
                    _df.set_index(['Time'], inplace=True)
                    _df.rename(index=_datetime_dict, inplace=True)
                    _df = _df.reindex(_datetime_list)
                except:
                    _df = DataFrame({'%s' % index : np.nan }, index = _datetime_list)
                _df.fillna(method='ffill', inplace=True)
                Day_data.loc[get_date_by_str(date), :] = _df.iloc[-1]
                Minute_data = Minute_data.append(_df)
                Day_data.fillna(method='ffill', inplace=True)
                Minute_data.fillna(method='ffill', inplace=True)
            os.mkdir('%smerged_index/%s' % (path, index))
            Day_data.to_csv('%smerged_index/%s/day.csv' % (path, index))
            Minute_data.to_csv('%smerged_index/%s/minute.csv' % (path, index))
            INDEX_DICT[index] = (Day_data.copy(deep=True), Minute_data.copy(deep=True))

def gen_pair_records(start, end, pair, type='day', mp_index_list=None, mp_date_list=None, path='./'):
    '''
    start, end: '%Y%m%d', say '20170707'
    pair: (index1, index2)
    返回DAY_DATA或者MINUTE_DATA的DataFrame：
        index       index_1        index_2         Signal
        start       1000            2000            0.5
        ...
        end         500             5000            0.1
    如果某一产品在start开始几天内都没有数据，即认为它还未上市或数据不能得到， 返回None
    '''
    if mp_index_list:
        index_1, index_2 = pair
        if type == 'day':
            _data_1 = mp_index_list[index_1][0]
            _data_2 = mp_index_list[index_2][0]
            start_date = get_date_by_str(start)
            k = get_start_by_window(end, 2, mp_date_list)
            end_date = get_date_by_str(k)
            _df_1 = _data_1.loc[start_date:end_date].copy(deep=True)
            _df_2 = _data_2.loc[start_date:end_date].copy(deep=True)
        elif type == 'minute':
            _data_1 = mp_index_list[index_1][1]
            _data_2 = mp_index_list[index_2][1]
            start_date = get_date_by_str(start)
            end_date = get_date_by_str(end)
            _df_1 = _data_1.loc[start_date:end_date].copy(deep=True)
            _df_2 = _data_2.loc[start_date:end_date].copy(deep=True)
    else:
        if len(INDEX_DICT) == 0:
			gen_index_info(path)
        index_1, index_2 = pair
        if type == 'day':
            _data_1 = INDEX_DICT[index_1][0]
            _data_2 = INDEX_DICT[index_2][0]
            start_date = get_date_by_str(start)
            end_date = get_date_by_str(get_start_by_window(end,2))
            _df_1 = _data_1.loc[start_date:end_date].copy(deep=True)
            _df_2 = _data_2.loc[start_date:end_date].copy(deep=True)
        elif type == 'minute':
            _data_1 = INDEX_DICT[index_1][1]
            _data_2 = INDEX_DICT[index_2][1]
            start_date = get_date_by_str(start)
            end_date = get_date_by_str(end)
            _df_1 = _data_1.loc[start_date:end_date].copy(deep=True)
            _df_2 = _data_2.loc[start_date:end_date].copy(deep=True)
    df = pd.concat([_df_1, _df_2], axis=1)
    if df.isnull().any().any():
        return None
	#记录每日收盘的指数，用于计算MA
    df['Signal'] = df[index_1] / df[index_2]
    return df

#主要回测函数
def get_good_portfolio(test_info, method='P&L'):
    '''
    通过迭代，得到该时间段内各产品对的参数结果
    test_info: DataFrame, 如下：
         Pair |  Parameter   |  Strategy  |   Mean  |  Volatility  |  Sharpe_Ratio  |  P&L  |
    0   'rb_j' 'W:20,T:0.20%'    'day'       10.34      120.67          1.56          1238.2
    ...
    根据不同的筛选标准，得到最佳组合：
    portfolio: set( ('Pair', 'Strategy', 'Parameter') )
    '''
    _portfolio = set()
    if method == 'P&L':
        best_para = DataFrame(np.nan, index=[], columns=['Parameter', 'Strategy', 'Sharpe_Ratio', 'P&L'])
        pairs_groups = test_info.groupby('Pair')
        for pair_str, group in pairs_groups:
            sort_by_Shp = group.sort_values(by='P&L', ascending=False)
            best_ = sort_by_Shp.iloc[0]
            best_para.loc['%s' % pair_str] = best_['Parameter'], best_['Strategy'], best_['Sharpe_Ratio'], best_['P&L']
            best_para.sort_values(by='P&L', ascending=False, inplace=True)
        for i in range(SCREENED_SIZE):
            _portfolio.add((best_para.iloc[i].name, best_para.iloc[i, 1], best_para.iloc[i, 0]))
    elif method == 'Sharpe':
        best_para = DataFrame(np.nan, index=[], columns=['Parameter', 'Strategy', 'Sharpe_Ratio', 'P&L'])
        pairs_groups = test_info.groupby('Pair')
        for pair_str, group in pairs_groups:
            sort_by_Shp = group.sort_values(by='Sharpe_Ratio', ascending=False)
            best_ = sort_by_Shp.iloc[0]
            best_para.loc['%s' % pair_str] = best_['Parameter'], best_['Strategy'], best_['Sharpe_Ratio'], best_['P&L']
            best_para.sort_values(by='Sharpe_Ratio', ascending=False, inplace=True)
        for i in range(SCREENED_SIZE):
            _portfolio.add((best_para.iloc[i].name, best_para.iloc[i, 1], best_para.iloc[i, 0]))
    return _portfolio

def screen_pairs(start, end, path='./' ):
    '''
    选择最近时间内表现最佳的100个产品对
    返回产品对的集合: set('index1_index2')
    '''
    Best_pairs = set()#denote the best pairs set
    start_date = get_date_by_str(start)
    pair_list = iter_all_products()
    window_list = [10, 20, 30, 40, 50]
    threshold_list = [0, 0.001, 0.002, 0.004, 0.006, 0.008]
    day_results = DataFrame(np.nan, index=[], columns=['Pair', 'Parameters', 'Sharpe_Ratio', 'Total_Profit'])
    _adj_start = get_start_by_window(start, 60)
    for pair in pair_list:
        _day_data = gen_pair_records(_adj_start, end, pair, 'day')
        if _day_data is None:#One of indexes is unavailable, skip the pair
            continue
        for w in window_list:
            __day_data = _day_data.copy(deep=True)
            __day_data['Signal_MA'] = __day_data['Signal'].rolling(window=w).mean()
            assert __day_data.loc[start_date:, 'Signal_MA'].isnull().any() == False, 'Day data reading error: incorrect window!'
            for t in threshold_list:
                traded_data = fast_day_processing(__day_data.loc[start_date:].copy(deep=True), w, t)
                sharpe_ratio = traded_data['Daily_Profit'].mean() / traded_data['Daily_Profit'].std() * (250 ** 0.5)
                tot_earning = traded_data['Cumul_Profit'].iloc[-1]
                info = {'Pair': '%s_%s' % pair, 'Parameters': 'W:%d,T:%.2f%%' % (w, t * 100), \
                        'Sharpe_Ratio': sharpe_ratio, 'Total_Profit': tot_earning}
                day_results = day_results.append(info, ignore_index=True)
    #根据夏普比率和总盈利最高各选择约80个，取交集
    sharpe_max_ = day_results.groupby(['Pair'])['Sharpe_Ratio'].max().sort_values(inplace=False, ascending=False)
    profit_max_ = day_results.groupby(['Pair'])['Total_Profit'].max().sort_values(inplace=False, ascending=False)
    sharpe_pairs = set(sharpe_max_.index[:80])
    profit_pairs = set(profit_max_.index[:80])
    Best_pairs = sharpe_pairs & profit_pairs
    #根据夏普比率和总盈利前三的平均值最高者中选择剩下的候选者
    SHP_MEAN, SHP_STD = day_results['Sharpe_Ratio'].mean(), day_results['Sharpe_Ratio'].std()
    PRF_MEAN, PRF_STD = day_results['Total_Profit'].mean(), day_results['Total_Profit'].std()
    mean_pairs = []
    for name, group in day_results.groupby(['Pair']):
        if name in Best_pairs:
            continue
        sorted_by_shp = group.sort_values(by='Sharpe_Ratio', ascending=False)
        shp_score = (sorted_by_shp.iloc[0:3, 2].mean() - SHP_MEAN) / SHP_STD
        sorted_by_prf = group.sort_values(by='Total_Profit', ascending=False)
        prf_score = (sorted_by_prf.iloc[0:3, 3].mean() - PRF_MEAN) / PRF_STD
        mean_pairs.append((name, prf_score + shp_score))
    mean_pairs.sort(key=lambda x: x[1], reverse=True)
    for item in mean_pairs:
        Best_pairs.add(item[0])
        if len(Best_pairs) >= 84:
            break
    return Best_pairs

def multi_task(para):
	#unpack the parameters
    start, end, pairs, para_dict, index_dict, date_list = para
    print 'Start process:%s, %d pairs, around%d parameters' % (os.getpid(), len(pairs), 50 * len(pairs))

    test_info = DataFrame(np.nan, \
                        index=[], \
                        columns=['Pair', 'Parameter', 'Strategy', 'Mean', 'Volatility', 'Sharpe_Ratio', 'P&L'])
    train_data = DataFrame(np.nan, \
                        index=[], \
                        columns=['Pair', 'Parameter', 'Strategy', 'Mean_1', 'Mean_2', 'Mean_3', 'Mean_4', 'Mean_5', 'Mean_6', \
                                    'Vol_1', 'Vol_2', 'Vol_3', 'Vol_4', 'Vol_5', 'Vol_6', 'Mean_y', 'Vol_y', 'Y'])

    mon_tuple = split_period(start)
    start_date = get_date_by_str(start)
    ADJ_start = get_start_by_window(start, 60, date_list)
    for pair_str in pairs:
        try:
            os.mkdir('./trade_record/%s-%s/%s/' % (start, end, pair_str))
        except:
            pass
        _pair = tuple(re.findall('[a-zA-Z]+', pair_str))
        _good_ma = para_dict[pair_str]
        window_list = get_windows(list(_good_ma))
        threshold_list = np.linspace(0, 0.01, 5)
        #load the index data
        day_record = gen_pair_records(ADJ_start, end, _pair, 'day', index_dict, date_list)
        minute_record = gen_pair_records(start, end, _pair, 'minute', index_dict, date_list)

        for w in map(int, window_list):
            #dump the data by the window
			_day_record = day_record.copy(deep=True)
			_day_record['Signal_MA'] = day_record['Signal'].rolling(window=w).mean()
			_day_record = _day_record.loc[start_date:]
			assert not _day_record.isnull().any().any(), 'Day records ma Error!'

			_min_daily_record = day_record.copy(deep=True)
			_min_daily_record['MA_t-1'] = _min_daily_record['Signal'].rolling(w).apply(lambda x: (x[:-1].sum()) / (w - 1))
			_min_daily_record = _min_daily_record.loc[start_date:]
			assert not _min_daily_record.isnull().any().any(), 'Minute records ma Error!'

			_minute_record = minute_record.copy(deep=True)

			for t in threshold_list:
                #back trading test
				day_traded = fast_day_processing(_day_record.copy(deep=True), w, t)
				minute_daily, _ = fast_minute_processing(_min_daily_record.copy(deep=True), _minute_record.copy(deep=True), w, t)

				#统计表现指标-前六个月
				train_end_date = get_date_by_str(mon_tuple[-1])

				day_mean = day_traded.loc[:train_end_date, 'Daily_Profit'].mean()
				day_vol = day_traded.loc[:train_end_date, 'Daily_Profit'].std()
				day_sharpe = day_mean / day_vol * (250 ** 0.5)
				day_pl = day_traded.loc[:train_end_date, 'Cumul_Profit'].iloc[-1]

				minute_mean = minute_daily.loc[:train_end_date, 'Daily_Profit'].mean()
				minute_vol = minute_daily.loc[:train_end_date, 'Daily_Profit'].std()
				minute_sharpe = minute_mean / minute_vol * (250 ** 0.5)
				minute_pl = minute_daily.loc[:train_end_date, 'Cumul_Profit'].iloc[-1]
				#记录回测结果和交易记录
				day_info = {'Pair': '%s_%s' % _pair, 'Parameter': 'W:%d,T:%.2f%%' % (w, t*100), \
							'Strategy': 'day', 'Mean': day_mean, 'Volatility': day_vol, 'Sharpe_Ratio': day_sharpe, 'P&L': day_pl}
				minute_info = {'Pair': '%s_%s' % _pair, 'Parameter': 'W:%d,T:%.2f%%' % (w, t*100), \
							'Strategy': 'minute', 'Mean': minute_mean, 'Volatility': minute_vol, 'Sharpe_Ratio': minute_sharpe, 'P&L': minute_pl}
				test_info = test_info.append(day_info, ignore_index=True)
				test_info = test_info.append(minute_info, ignore_index=True)

				#minute_traded.to_csv('./trade_record/%s/%s_%s/minute_W%d_T%.2f%%.csv' % ((start+'-'+end), _pair[0], _pair[1], w, 100.0 * t))
				minute_daily.to_csv('./trade_record/%s/%s_%s/minute_profit_W%d_T%.2f%%.csv' % ((start+'-'+end), _pair[0], _pair[1], w, 100.0 * t))
				day_traded.to_csv('./trade_record/%s/%s_%s/daily_W%d_T%.2f%%.csv' % ((start+'-'+end), _pair[0], _pair[1], w, 100.0 * t))
				#记录每月的交易记录
				day_monthly = []
				min_monthly = []
				for mon in range(6):
					train_start_ = get_date_by_str(mon_tuple[mon])
					train_end_date_ = get_date_by_str(mon_tuple[mon+1])

					day_mean = day_traded.loc[train_start_: train_end_date_, 'Daily_Profit'].mean()
					day_vol = day_traded.loc[train_start_: train_end_date_, 'Daily_Profit'].std()
					day_monthly.append(tuple((day_mean, day_vol)))
					min_mean = minute_daily.loc[train_start_: train_end_date_, 'Daily_Profit'].mean()
					min_vol = minute_daily.loc[train_start_: train_end_date_, 'Daily_Profit'].std()
					min_monthly.append(tuple((day_mean, day_vol)))

				pred_day_mean = day_traded.loc[train_end_date:, 'Daily_Profit'].mean()
				pred_day_vol = day_traded.loc[train_end_date:, 'Daily_Profit'].std()
				pred_min_mean = minute_daily.loc[train_end_date:, 'Daily_Profit'].mean()
				pred_min_vol = minute_daily.loc[train_end_date:, 'Daily_Profit'].std()
				day_Y = next_performance(day_mean, day_sharpe, pred_day_mean, pred_day_vol)
				min_Y = next_performance(minute_mean, minute_sharpe, pred_min_mean, pred_min_vol)
				train_day = {'Pair': '%s_%s' % _pair, 'Parameter': 'W:%d,T:%.2f%%' % (w, t*100), 'Strategy': 'day', \
							'Mean_1': day_monthly[0][0], 'Mean_2': day_monthly[1][0], 'Mean_3': day_monthly[2][0], \
							'Mean_4': day_monthly[3][0], 'Mean_5': day_monthly[4][0], 'Mean_6': day_monthly[5][0], \
							'Vol_1': day_monthly[0][1], 'Vol_2': day_monthly[1][1], 'Vol_3': day_monthly[2][1], \
							'Vol_4': day_monthly[3][1], 'Vol_5': day_monthly[4][1], 'Vol_6': day_monthly[5][1], \
							'Mean_y': pred_day_mean, 'Vol_y': pred_day_vol, 'Y': day_Y}
				train_min = {'Pair': '%s_%s' % _pair, 'Parameter': 'W:%d,T:%.2f%%' % (w, t*100), 'Strategy': 'minute', \
							'Mean_1': min_monthly[0][0], 'Mean_2': min_monthly[1][0], 'Mean_3': min_monthly[2][0], \
							'Mean_4': min_monthly[3][0], 'Mean_5': min_monthly[4][0], 'Mean_6': min_monthly[5][0], \
							'Vol_1': min_monthly[0][1], 'Vol_2': min_monthly[1][1], 'Vol_3': min_monthly[2][1], \
							'Vol_4': min_monthly[3][1], 'Vol_5': min_monthly[4][1], 'Vol_6': min_monthly[5][1], \
							'Mean_y': pred_min_mean, 'Vol_y': pred_min_vol, 'Y': min_Y}
				train_data = train_data.append(train_day, ignore_index=True)
				train_data = train_data.append(train_min, ignore_index=True)
    return test_info, train_data

def multi_screen_paraANDstra(start, end, path='./'):
    if len(INDEX_DICT) == 0:
		gen_index_info(path)
    try:
        os.mkdir('./trade_record/%s-%s/' % (start, end))
    except:
        pass
    mon_tuple = split_period(start)
    train_end = mon_tuple[-1]

    best_pairs = tuple(screen_pairs(start, train_end, path))
    q_len = len(best_pairs) / 7
    seven_parts = [best_pairs[q_len * i: q_len * (i+1)] for i in range(7)]

    pair_paras = [get_good_ma(start, train_end, part) for part in seven_parts]
    paras = [(start, end, pairs, para_dict, dict(INDEX_DICT), list(DATE_LIST)) for pairs, para_dict in zip(seven_parts, pair_paras)]

    print 'Start iteration by seven processing, now is the parent:%s' % os.getpid()
    test_train = []
    with ProcessPoolExecutor() as pool:
        for data in pool.map(multi_task, paras):
            test_train.append(data)
    test_info = DataFrame()
    train_data = DataFrame()

    for info in test_train:
		test_info = test_info.append(info[0], ignore_index=True)
		train_data =train_data.append(info[1], ignore_index=True)

    train_data.to_csv('./trade_record/%s/train_data.csv' % (start+'-'+end))
    test_info.to_csv('./trade_record/%s/test_info.csv' %  (start+'-'+end))

	#完成迭代，选择最好的20对产品， 和它们最好的参数
    Portfolio_pl = get_good_portfolio(test_info, 'P&L')
    Portfolio_sp = get_good_portfolio(test_info, 'Sharpe')
    return Portfolio_pl, Portfolio_sp

def back_testing(start, end, portfolio_pl=set(), portfolio_sp=set(), path='./'):
    init_datesANDtimesANDproducts(path)
    print 'backetesting from %s to %s...' % (start, end)
    duration_mons = delta_mon(start, end)
    portfolio = {'P&L': portfolio_pl, 'Sharpe': portfolio_sp}
    try:
        os.mkdir('%sbacktest_records/' % path)
    except:
        pass

    for mon in range(duration_mons):
        next_start = get_next_mon(start, mon)
        next_end = get_next_mon(next_start, 1)
        backtest_start = get_next_mon(next_start, -6)
        print "Start backtesting during %s - %s" % (backtest_start, next_start)
        screened_portfolio_pl, screened_portfolio_sp = multi_screen_paraANDstra(backtest_start, next_end, path)#portfolio: set( ('Pair', 'Strategy', 'Parameter') )
        screened_portfolio = {'P&L': screened_portfolio_pl, 'Sharpe': screened_portfolio_sp}


        new_portfolio_pl = set(x[0] for x in screened_portfolio_pl)
        new_portfolio_sp = set(x[0] for x in screened_portfolio_sp)
        new_portfolio = {'P&L': new_portfolio_pl, 'Sharpe': new_portfolio_sp}
        try:
            os.mkdir('%sbacktest_records/%s-%s/' % (path, next_start, next_end))
        except:
            pass

        for method in ['P&L', 'Sharpe']:
            remaining_pairs = portfolio[method] & new_portfolio[method]
            new_pairs = new_portfolio[method] - portfolio[method]
            os.mkdir('%sbacktest_records/%s-%s/%s/' % (path, next_start, next_end, method))

            screened_info = DataFrame(columns=['Pair', 'Strategy', 'Parameter'])

            for pair_str, stra, para in screened_portfolio[method]:
                pair = tuple(re.findall('[a-zA-Z]+', pair_str))
                window, threshold = tuple(re.findall(':([0-9\.]+)', para))
                window = int(window)
                threshold = float(threshold) / 100.0

                start_date = get_date_by_str(next_start)
                adj_start = get_start_by_window(next_start, window)

                day_data = gen_pair_records(adj_start, next_end, pair, 'day')
                min_daily_data = day_data.copy(deep=True)

                day_data['Signal_MA'] = day_data['Signal'].rolling(window=window).mean()
                day_data = day_data.loc[start_date:]
                assert not day_data.isnull().any().any(), 'Day records ma Error!'

                min_daily_data['MA_t-1'] = min_daily_data['Signal'].rolling(window).apply(lambda x: (x[:-1].sum()) / (window - 1))
                min_daily_data = min_daily_data.loc[start_date:]
                assert not min_daily_data.isnull().any().any(), 'Minute records ma Error!'

                minute_data = gen_pair_records(next_start, next_end, pair, 'minute')


                if pair_str in remaining_pairs:
                    pre_record = pd.read_csv('%sbacktest_records/%s-%s/%s/%s.csv' % (path, get_next_mon(start, mon-1), get_next_mon(start, mon), method, pair_str), \
                                                header=0 )
                    price_1 = pre_record['%s' % pair[0]].iloc[-1]
                    price_2 = pre_record['%s' % pair[1]].iloc[-1]
                    position_1 = pre_record['Position_1'].iloc[-1]
                    position_2 = pre_record['Position_2'].iloc[-1]
                    if stra == 'day':
                        next_traded = fast_day_processing(day_data, window, threshold, True, price_1, price_2, position_1, position_2)
                    elif stra == 'minute':
                        next_daily, next_traded = fast_minute_processing(min_daily_data, minute_data, window, threshold, True, price_1, price_2, position_1, position_2)
                else:
                    if stra == 'day':
                        next_traded = fast_day_processing(day_data, window, threshold)
                    elif stra == 'minute':
                        next_daily, next_traded = fast_minute_processing(min_daily_data, minute_data, window, threshold)
                next_traded.to_csv('%sbacktest_records/%s-%s/%s/%s.csv' % (path, next_start, next_end, method, pair_str))
                if stra == 'day':
                    trade_data = next_traded
                elif stra == 'minute':
                    trade_data = next_daily
                mean = trade_data['Daily_Profit'].mean()
                vol = trade_data['Daily_Profit'].std()
                sharpe = mean / vol * (250 ** 0.5)
                pnl = trade_data['Cumul_Profit'].iloc[-1]
                info = {'Pair' : pair_str, 'Strategy': stra, 'Parameter': para, 'Mean': mean, 'Volatility': vol, 'Sharpe_Ratio': sharpe, 'P&L': pnl}
                screened_info = screened_info.append(info, ignore_index=True)
            screened_info.to_csv('%sbacktest_records/%s-%s/%s/para_info.csv' % (path, next_start, next_end, method))
            portfolio[method] = new_portfolio[method]



if __name__ == '__main__':
    try:
        os.mkdir('./trade_record')
    except:
        pass
    portfolio_pl = set()
    portfolio_sp = set()
    para_pl = pd.read_csv('./backtest_records/20170301-20170401/P&L/para_info.csv', header=0)
    para_sp = pd.read_csv('./backtest_records/20170301-20170401/Sharpe/para_info.csv', header=0)
    for pair_pl, pair_sp in zip(para_pl['Pair'], para_sp['Pair']):
        portfolio_pl.add(pair_pl)
        portfolio_sp.add(pair_sp)
    print portfolio_pl
    print 'Start back testing the CTA from 20160601-20170701...'
    back_testing('20170401', '20170701', portfolio_pl, portfolio_sp)
