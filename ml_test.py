#coding:utf-8
from sklearn import linear_model
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler

from mulproc_backtesting_cta import *

def preprocessing_df(data_i):
        data_i['Mean_Tot'] = data_i[['Mean_1', 'Mean_2', 'Mean_3', 'Mean_4', 'Mean_5', 'Mean_6']].sum(axis=1) / 6
    	data_i.drop_duplicates(['Pair', 'Mean_Tot'], inplace=True)
    	data_i['Window'] = data_i['Parameter'].apply(lambda x: round(int(re.findall('[0-9]+', x)[0]) / 10.0) * 10)
    	data_i['Threshold(%)'] = data_i['Parameter'].apply(lambda x: float(re.findall('[0-9\.]+', x)[1]))
    	data_i['Strategy'] = data_i['Strategy'].map({'day': 0.0, 'minute': 1.0})
        data_i['Y'] = data_i.apply(lambda x: 1 if x['Mean_y'] > 0.6 * x['Mean_Tot'] else -1, axis=1)
    	data_i = data_i.reindex_axis(['Pair', 'Parameter', 'Strategy', 'Window', 'Threshold(%)', 'Mean_Tot', 'Mean_1', 'Mean_2', 'Mean_3', \
    						'Mean_4', 'Mean_5', 'Mean_6', 'Vol_1', 'Vol_2', 'Vol_3', 'Vol_4', 'Vol_5', 'Vol_6', \
    						'Mean_y', 'Y'], axis=1)
        return data_i

def train_coef(train_data):
    train_data = train_data.iloc[np.random.permutation(len(train_data))]
    X = train_data.iloc[:, 2:-2].as_matrix()
    Y_class = np.array(train_data['Y']).astype(np.int8)
    scaler = StandardScaler()
    scaler.fit(X)
    X_scaled = scaler.transform(X)

    clf = SVC(C=100.0)
    clf.fit(X_scaled, Y_class)
    return scaler, clf

def get_good_portfolio(next_start, next_end, test_info):
    '''
    以前6个月的train_data训练SVM
    将每个配对中回测最佳的3组参数，按P&L排序
    根据得到的模型clf，判断是否能够达到60%,如果可以就选中
    '''
    bt_start = get_next_mon(next_start, -6)
    data = DataFrame(columns=['Pair', 'Parameter', 'Strategy', 'Window', 'Threshold(%)', 'Mean_Tot', 'Mean_1', 'Mean_2', 'Mean_3', \
                        'Mean_4', 'Mean_5', 'Mean_6', 'Vol_1', 'Vol_2', 'Vol_3', 'Vol_4', 'Vol_5', 'Vol_6', \
                        'Mean_y', 'Y'])
    for i in range(6):
        data_i = pd.read_csv('./trade_record/%s-%s/train_data.csv' % (get_next_mon(bt_start, i-6), get_next_mon(bt_start, i+1)), header=0, index_col=0)
        data = data.append(preprocessing_df(data_i), ignore_index=True)
    print 'Training the SVC model during %s-%s' % (get_next_mon(bt_start, -6), get_next_mon(bt_start, 6))
    scaler, clf = train_coef(data)

    fore_data = pd.read_csv('./trade_record/%s-%s/train_data.csv' % (get_next_mon(next_start, -6), next_end), header=0, index_col=0)
    fore_data = preprocessing_df(fore_data)

    _portfolio = set()
    _pair = set()
    best_para = DataFrame(np.nan, index=[], columns=['Parameter', 'Strategy', 'Sharpe_Ratio', 'P&L'])
    for pair_str, group in test_info.groupby('Pair'):
        sort_by_pnl = group.sort_values(by='P&L', ascending=False)
        best_ = sort_by_pnl.iloc[0:3]
        df = best_[['Pair', 'Parameter', 'Strategy', 'Sharpe_Ratio', 'P&L']]
        best_para = best_para.append(df, ignore_index=True)
    best_para.sort_values(by='P&L', ascending=False, inplace=True)

    stra_dict = {'minute': 1.0, 'day': 0.0}
    for info in best_para.itertuples():
        forecast_data = fore_data.loc[(fore_data['Strategy'] == stra_dict[info.Strategy]) & (fore_data['Pair'] == info.Pair) & (fore_data['Parameter'] == info.Parameter)]
        pred_X = forecast_data.iloc[:,2:-2].as_matrix()
        scaled_pred_X = scaler.transform(pred_X)
        pred_y = clf.predict(scaled_pred_X)

        if (pred_y == 1) and (not info.Pair in _pair):
            _portfolio.add((info.Pair, info.Strategy, info.Parameter))
            _pair.add(info.Pair)
        if len(_portfolio) >= SCREENED_SIZE:
            break
    if len(_portfolio) < SCREENED_SIZE:
        for info in best_para.itertuples():
            if (not info.Pair in _pair):
                _portfolio.add((info.Pair, info.Strategy, info.Parameter))
                _pair.add(info.Pair)
            if len(_portfolio) >= SCREENED_SIZE:
                break

    return _portfolio

def back_testing(start, end, portfolio=set(), path='./'):
    init_datesANDtimesANDproducts(path)
    print 'backetesting from %s to %s...' % (start, end)
    duration_mons = delta_mon(start, end)
    try:
        os.mkdir('%sbacktest_records/' % path)
    except:
        pass

    for mon in range(duration_mons):
        next_start = get_next_mon(start, mon)
        next_end = get_next_mon(next_start, 1)
        backtest_start = get_next_mon(next_start, -6)
        print "Start backtesting during %s - %s" % (backtest_start, next_start)
        test_info = pd.read_csv('./trade_record/%s-%s/test_info.csv' % (get_next_mon(next_start, -6), next_end), header=0, index_col=0)
        screened_portfolio = get_good_portfolio(next_start, next_end, test_info)


        new_portfolio = set(x[0] for x in screened_portfolio)
        try:
            os.mkdir('%sbacktest_records/%s-%s/' % (path, next_start, next_end))
        except:
            pass
        remaining_pairs = portfolio & new_portfolio
        screened_info = DataFrame(columns=['Pair', 'Strategy', 'Parameter'])

        for pair_str, stra, para in screened_portfolio:
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
                pre_record = pd.read_csv('%sbacktest_records/%s-%s/%s.csv' % (path, get_next_mon(start, mon-1), get_next_mon(start, mon), pair_str), \
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
            next_traded.to_csv('%sbacktest_records/%s-%s/%s.csv' % (path, next_start, next_end, pair_str))
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
        screened_info.to_csv('%sbacktest_records/%s-%s/para_info.csv' % (path, next_start, next_end))
        portfolio = new_portfolio

if __name__ == '__main__':
    try:
        os.mkdir('./trade_record')
    except:
        pass
    Portfolio = set()
    para_pl = pd.read_csv('./backtest_records/20161201-20170101/para_info.csv', header=0)
    for pair in para_pl['Pair']:
        Portfolio.add(pair)
    print Portfolio
    print 'Start back testing the CTA from 20170101-20170701...'
    back_testing('20170101', '20170701', Portfolio)
