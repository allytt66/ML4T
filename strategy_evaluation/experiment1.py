# Experiment 1 should compare the results of your manual strategy and the strategy learner. It should: 

# Compare your Manual Strategy with your Strategy Learner in-sample trading JPM. Create a chart that shows:  
# Value of the ManualStrategy portfolio (normalized to 1.0 at the start)  
# Value of the StrategyLearner portfolio (normalized to 1.0 at the start)  
# Value of the Benchmark portfolio (normalized to 1.0 at the start)  

# Compare your Manual Strategy with your Strategy Learner out-of-sample trading JPM. Create a chart that shows:  
# Value of the ManualStrategy portfolio (normalized to 1.0 at the start)  
# Value of the StrategyLearner portfolio (normalized to 1.0 at the start)  
# Value of the Benchmark portfolio (normalized to 1.0 at the start) 

from StrategyLearner import StrategyLearner 
from  ManualStrategy import ManualStrategy
from marketsimcode import compute_portvals 
import util as ut  	

import datetime as dt 
import pandas as pd
import matplotlib.pyplot as plt 

def author():
    return "yzhang3946"

def convert_trade(trades, symbol): 
    
    df_trade_strategy = pd.DataFrame(columns=['Date', 'Symbol', 'Order','Shares'])

    for i in range(len(trades)): 

        if trades.iloc[i][0] > 0.0: 
            data = {'Date': trades.index[i], 
                    'Symbol': symbol, 
                    'Order': "BUY", 
                    'Shares': abs(trades.iloc[i][0])}
            df_trade_strategy = df_trade_strategy.append(data, ignore_index=True)
            
        elif trades.iloc[i][0] < 0.0: 
            data = {'Date': trades.index[i], 
                    'Symbol': symbol, 
                    'Order': "SELL", 
                    'Shares': abs(trades.iloc[i][0])}
            
            df_trade_strategy = df_trade_strategy.append(data, ignore_index=True)
    
    return df_trade_strategy


def experiment1(): 
    symbol = 'JPM'
    sd=dt.datetime(2008, 1, 1)		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    ed=dt.datetime(2009, 12, 31)  

    tsd=dt.datetime(2010, 1, 1)		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    ted=dt.datetime(2011, 12, 31)  

    sv = 100000
    
    prices_insample =  ut.get_data([symbol], pd.date_range(sd, ed))[[symbol]]
    prices_outsample =  ut.get_data([symbol], pd.date_range(tsd, ted))[[symbol]]

    print('start in sample')


### in sample ###
    ms = ManualStrategy()
    manual_trades = ms.testPolicy(symbol=symbol, sd=sd, ed=ed, sv=sv)
    manual_insample = compute_portvals(manual_trades, start_val=sv, commission=0, impact = 0.000, start=sd, end=ed)

    print('start q learner')
 
    learner = StrategyLearner(verbose = False, commission = 0.0, impact = 0.000)
    learner.add_evidence(symbol=symbol, sd=sd, ed=ed, sv=sv)

    print('start test q learner')

    learner_trades = learner.testPolicy(symbol=symbol, sd=sd, ed=ed, sv=sv)
    learner_trades = convert_trade(learner_trades, symbol)
    learner_insample = compute_portvals(learner_trades, start_val = sv, commission=0, impact=0.000, start=sd, end=ed)

    bench_trades = pd.DataFrame(columns=['Date', 'Symbol', 'Order','Shares'])
    data = {'Date': prices_insample.index[0], 
        'Symbol': symbol, 
        'Order':'BUY', 
        'Shares':1000}
    bench_trades = bench_trades.append(data, ignore_index=True)
    bench_insample = compute_portvals(bench_trades, start_val = sv, commission=0, impact=0.000, start=sd, end=ed)

    print('start plot')


    manual_insample = manual_insample/ manual_insample[0]
    learner_insample = learner_insample/learner_insample[0]
    bench_insample = bench_insample/bench_insample[0]

    fig, ax = plt.subplots()
    ax.plot(manual_insample, label='ManualStrategy')
    ax.plot(learner_insample, label="StrategyLearner")
    ax.plot(bench_insample, label="Benchmark")
    ax.set_ylabel('normalized cumulative return')
    ax.set_title('experiment 1 in sample')
    fig.set_figwidth(14)
    ax.legend()
    ax.grid()
    fig.savefig('exp1_insample.png')

    print('start out sample')


### out sample ### 
    manual = ManualStrategy()
    manual_trades = manual.testPolicy(symbol=symbol, sd=tsd, ed=ted, sv=sv)
    manual_outsample = compute_portvals(manual_trades, start_val=sv, commission=0, impact = 0.000, start=tsd, end=ted)
 
    learner_trades = learner.testPolicy(symbol=symbol, sd=tsd, ed=ted, sv=sv)
    learner_trades = convert_trade(learner_trades, symbol)
    learner_outsample = compute_portvals(learner_trades, start_val = sv, commission=0, impact=0.000, start=tsd, end=ted)

    bench_trades = pd.DataFrame(columns=['Date', 'Symbol', 'Order','Shares'])
    data = {'Date': prices_outsample.index[0], 
        'Symbol': "JPM", 
        'Order':'BUY', 
        'Shares':1000}
    bench_trades = bench_trades.append(data, ignore_index=True)
    bench_outsample = compute_portvals(bench_trades, start_val = sv, commission=0, impact=0.000, start=tsd, end=ted)

 
    manual_outsample = manual_outsample / manual_outsample[0]
    learner_outsample = learner_outsample/learner_outsample[0]
    bench_outsample = bench_outsample/bench_outsample[0]

    fig, ax = plt.subplots()
    ax.plot(manual_outsample, label='ManualStrategy')
    ax.plot(learner_outsample, label="StrategyLearner")
    ax.plot(bench_outsample, label="Benchmark")
    ax.set_ylabel('normalized cumulative return')
    ax.set_title('experiment 1 out sample')
    fig.set_figwidth(14)
    ax.legend()
    ax.grid()

    fig.savefig('exp1_outsample.png')

if __name__ == "__main__": 
    experiment1()