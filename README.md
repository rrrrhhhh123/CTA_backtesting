#coding:utf-8
#CTA趋势跟踪策略 推进回测报告#

multiproc_backtesting_cta.py
回测时间段：20160401-20170701
组合大小：每个月15个期货产品配对， 每个产品配对用10,000本金交易，共15万本金
回测效果：利用P&L筛选组合得到15个月累积收益约58,600
		 利用Sharpe筛选组合得到15个月累计收益为-5,200
回测步骤：
20160401-20160501：
1. 读取前6个月数据，利用日收盘交易策略进行遍历参数，选取出最佳的80个，选取规则如下：
	a)每个产品配对分别记录P&L最优的参数及其P&L, 和Sharpe最优的参数及其Sharpe, 将两个序列分别排序, 对其各前80进行交集。
	b)每个产品配对记录其P&L最优的前三个P&L的平均值，Sharpe同理，对两个平均值均一化，等权求和记作Score，对Score进行排序，从高到低选择产品配对，补足80个为止。
	注: 参数遍历具体为：MA:10, 20, 30, 40, 50; T:0, 0.1%, 0.2%, 0.4%, 0.6%, 0.8%， 共30种参数
		一般a)可以选出50-60个

2. 选取出的80个产品配对，分别取出对应的前六个月回测Sharpe最佳的3种参数的MA值，记作good_ma
	对这些产品配对进行前六个月的回测，参数设定为good_ma附近（5-9个ma，因为good_ma可能是1-3个），T为0.00%, 0.25%, 0.50%, 0.75%, 0.10%
	根据特定的选取规则，得到15个产品配对，进入下一个月的交易：
		a)如果某产品配对不属于上一个月的产品组合中，则第一天/第一分钟不考虑threshold直接进入，后面正常交易
		b)如果某产品配对属于上一个月的产品组合种，则保持其持仓量，正常交易
	注：选取规则：
		1.Sharpe最优：以前六个月回测Sharpe最高的15个产品配对为新的组合
		2.P&L最优：以前六个月P&L最高的15个产品配对为新的组合

3. 进入下一个月的回测。

注：在1.中实际读取了前6个月和下一个月，用作训练数据，所有涉及产品选择和参数选择的步骤均为偷看下一个月的数据。

结论：
	1.产品配对选择比参数选择重要
	2.Day和Minute并无绝对的优劣性（但未统计）
	3.利用P&L作为选取指标从16年-17年的回测来看远优于Sharpe，但Sharpe回测每个月的P&L的波动的确较小

Future Work：
	1.考虑是否可以从前六个月的收益和波动等信息中预测下个月的收益
		清理数据
		收益、波动、策略作为特征
		回归：预测下个月的P&L，（或者排名）
		分类：预测下个月表现是否更好
	如果看预测效果还行，考虑根据模型重新回测交易，看一下与平凡的筛选的区别
	2.在选取组合的时候，是否不绝对选取最优的， 例如在历史结果相近的情况下，优先选取上一个月组合中有的产品配对，可以在1.的比较中加入该实验。

#SVM分类学习优化配对选择#
ml_test.py
首先对20170401-20170701的回测数据进行训练，希望能够找到合适的模型和参数
1. Lasso Regression：对已有的12个feature仅有0.12的R2
2. Logistic Regression: 不添加多次项的情况F1score仅0.10
3. SVM: kernel取rbf，F1 Score为0.90，不删除特征，C选100最佳
对20170101-20170701回测：
利用之前一年的回测数据训练SVM，得到的模型clf
对上6个月的数据进行回测，根据P&L排序，从前往后用clf预测其是否能够超过上六个月表现的60%
预测结果为1才将其选入
结果：
		P&L(ML)		P&L(pnl)
2017-01 -24050.2	-29587.6
2017-02 -7183.19	-872.511
2017-03   2900.3	7374.49
2017-04 -1700.46	-2081.47
2017-05  -1212.6	1819.53
2017-06  6332.62	6465.74
从2017年的回测来看，利用SVM训练得到的模型来判断产品组合并没有太好的效果。
