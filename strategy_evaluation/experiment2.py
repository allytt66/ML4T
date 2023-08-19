# Conduct an experiment with your StrategyLearner that shows how changing the value of impact should affect in-sample trading behavior.

# Select two metrics, and generate tests that will provide you with at least 3 measurements when trading JPM on the in-sample period with a commission of $0.00. Generate charts that support your tests and show your results. 

# The code that implements this experiment and generates the relevant charts and data should be submitted as experiment2.py. 

# See the ‘Report’ section on Experiment 2 for more details. 

from StrategyLearner import StrategyLearner 
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


def get_stats(portvals):
    cr = portvals[-1] / portvals[0] - 1
    dr = (portvals / portvals.shift(1) - 1).iloc[1:]
    sddr = dr.std()
    adr = dr.mean()
    return cr, sddr, adr

def experiment2(): 

    symbol = 'JPM'
    sd=dt.datetime(2008, 1, 1)		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    ed=dt.datetime(2009, 12, 31)  

    tsd=dt.datetime(2010, 1, 1)		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    ted=dt.datetime(2011, 12, 31)  

    sv = 100000
    
    prices_insample =  ut.get_data([symbol], pd.date_range(sd, ed))[[symbol]]
    prices_outsample =  ut.get_data([symbol], pd.date_range(tsd, ted))[[symbol]]

    print('start in sample')

    sglearner0 = StrategyLearner(verbose = False, commission = 0.0, impact = 0.000)
    sglearner1 = StrategyLearner(verbose = False, commission = 0.0, impact = 0.010)
    sglearner2 = StrategyLearner(verbose = False, commission = 0.0, impact = 0.020)

    for i in [sglearner0, sglearner1, sglearner2]: 
        i.add_evidence(symbol=symbol, sd=sd, ed=ed, sv=sv)


    learner_trades0 = sglearner0.testPolicy(symbol=symbol, sd=sd, ed=ed, sv=sv)
    learner_trades0 = convert_trade(learner_trades0, symbol)
    learner_insample0 = compute_portvals(learner_trades0, start_val = sv, commission=0, impact=0.000, start=sd, end=ed)

    cr0, sddr0, adr0 = get_stats(learner_insample0)

    learner_trades1 = sglearner1.testPolicy(symbol=symbol, sd=sd, ed=ed, sv=sv)
    learner_trades1 = convert_trade(learner_trades1, symbol)
    learner_insample1 = compute_portvals(learner_trades1, start_val = sv, commission=0, impact=0.000, start=sd, end=ed)

    cr1, sddr1, adr1 = get_stats(learner_insample1)

    learner_trades2 = sglearner2.testPolicy(symbol=symbol, sd=sd, ed=ed, sv=sv)
    learner_trades2 = convert_trade(learner_trades2, symbol)
    learner_insample2 = compute_portvals(learner_trades2, start_val = sv, commission=0, impact=0.000, start=sd, end=ed)

    cr2, sddr2, adr2 = get_stats(learner_insample2)

    learner_insample0 = learner_insample0/ learner_insample0[0]
    learner_insample1 = learner_insample1/ learner_insample1[0]
    learner_insample2 = learner_insample2/ learner_insample2[0]

    fig, ax = plt.subplots()
    ax.plot(learner_insample0, label='No impact', alpha=0.7)
    ax.plot(learner_insample1, label="Small impact", alpha=0.7)
    ax.plot(learner_insample2, label="Big impact", alpha=0.7)
    ax.set_ylabel('normalized cumulative return')
    ax.set_title('experiment 2 in sample')
    fig.set_figwidth(14)
    ax.legend()
    ax.grid()
    fig.savefig('exp2_insample.png')

    print("no impact")
    print(f"cumulative return {cr0}")
    print(f"sddr {sddr0}")
    print(f"adr {adr0}")

    print("small impact")
    print(f"cumulative return {cr1}")
    print(f"sddr {sddr1}")
    print(f"adr {adr1}")


    print("small impact")
    print(f"cumulative return {cr2}")
    print(f"sddr {sddr2}")
    print(f"adr {adr2}")

if __name__ == "__main__": 
    experiment2()
