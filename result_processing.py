'''
Scripts to analysis and modify the results
'''
start = '20170101'
end = '20170701'
duration_mons = delta_mon(start, end)

mons_index = pd.period_range('201701', '201706', freq='M')
backtest_pl = DataFrame(columns=['P&L'], index=mons_index)
backtest_sp = DataFrame(columns=['P&L'], index=mons_index)
backtest_ml = DataFrame(columns=['P&L'], index=mons_index)

for mon in range(duration_mons):
    next_start = get_next_mon(start, mon)
    next_end = get_next_mon(next_start, 1)
    data_pl = pd.read_csv('./backtest_records_2/%s/P&L/para_info.csv'%(next_start+'-'+next_end), header=0, index_col=0)
    data_sp = pd.read_csv('./backtest_records_2/%s/Sharpe/para_info.csv'%(next_start+'-'+next_end), header=0, index_col=0)
    data_ml = pd.read_csv('./backtest_records/%s/para_info.csv'%(next_start+'-'+next_end), header=0, index_col=0)

    info_pl = data_pl['P&L'].sum()
    info_sp = data_sp['P&L'].sum()
    info_ml = data_ml['P&L'].sum()

    backtest_pl.loc[next_start, 'P&L'] = info_pl
    backtest_sp.loc[next_start, 'P&L'] = info_sp
    backtest_ml.loc[next_start, 'P&L'] = info_ml




def backtesting(start, end):
    duration_mons = delta_mon(start, end)
    portfolio_pl = set()
    portfolio_sp = set()
    portfolio = {'P&L': portfolio_pl, 'Sharpe': portfolio_sp}

    try:
        os.mkdir('backtest_records_2')
    except:
        pass

    for mon in range(duration_mons):
        next_start = get_next_mon(start, mon)
        next_end = get_next_mon(next_start, 1)
        para_info_pl = pd.read_csv('./backtest_records/%s-%s/P&L/para_info.csv' % (next_start, next_end), header=0, index_col=0)
        para_info_sp = pd.read_csv('./backtest_records/%s-%s/Sharpe/para_info.csv' % (next_start, next_end), header=0, index_col=0)
        screened_portfolio_pl = set()
        screened_portfolio_sp = set()
        for row_pl, row_sp in zip(para_info_pl.itertuples(), para_info_sp.itertuples()):
            info_pl = (row_pl.Pair, row_pl.Strategy, row_pl.Parameter)
            info_sp = (row_sp.Pair, row_sp.Strategy, row_sp.Parameter)
            screened_portfolio_pl.add(info_pl)
            screened_portfolio_sp.add(info_sp)
        screened_portfolio = {'P&L': screened_portfolio_pl, 'Sharpe': screened_portfolio_sp}

        new_portfolio_pl = set(x[0] for x in screened_portfolio_pl)
        new_portfolio_sp = set(x[0] for x in screened_portfolio_sp)
        new_portfolio = {'P&L': new_portfolio_pl, 'Sharpe': new_portfolio_sp}
        try:
            os.mkdir('./backtest_records_2/%s-%s/' % (next_start, next_end))
        except:
            pass

        for method in ['P&L', 'Sharpe']:
            remaining_pairs = portfolio[method] & new_portfolio[method]
            os.mkdir('./backtest_records_2/%s-%s/%s/' % (next_start, next_end, method))

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
                    pre_record = pd.read_csv('./backtest_records_2/%s-%s/%s/%s.csv' % (get_next_mon(start, mon-1), get_next_mon(start, mon), method, pair_str), \
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
                next_traded.to_csv('./backtest_records_2/%s-%s/%s/%s.csv' % (next_start, next_end, method, pair_str))
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
            screened_info.to_csv('./backtest_records_2/%s-%s/%s/para_info.csv' % (next_start, next_end, method))
            portfolio[method] = new_portfolio[method]


import matplotlib.pyplot as plt
fig = plt.figure()
ax1 = fig.add_subplot(2,1,1)
ax2 = fig.add_subplot(2,1,2)
ax1.plot(backtest_pl['P&L'], 'g-', label= 'P&L each month')
ax1.plot(backtest_pl.cumsum(), 'r--', label='Cumul_P&L')
ax1.set_title('P&L by pnl')
