""""""  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
"""MC1-P2: Optimize a portfolio.  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
Copyright 2018, Georgia Institute of Technology (Georgia Tech)  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
Atlanta, Georgia 30332  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
All Rights Reserved  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
Template code for CS 4646/7646  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
Georgia Tech asserts copyright ownership of this template and all derivative  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
works, including solutions to the projects assigned in this course. Students  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
and other users of this template code are advised not to share it with others  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
or to make it available on publicly viewable websites including repositories  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
such as github and gitlab.  This copyright statement should not be removed  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
or edited.  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
We do grant permission to share solutions privately with non-students such  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
as potential employers. However, sharing with other current or future  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
students of CS 7646 is prohibited and subject to being investigated as a  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
GT honor code violation.  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
-----do not edit anything above this line---  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
Student Name: Tucker Balch (replace with your name)  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
GT User ID: yzhang3946 (replace with your User ID)  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
GT ID: 903862041 (replace with your GT ID)  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
"""  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 


import datetime as dt  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
import numpy as np  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
import matplotlib.pyplot as plt  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
import pandas as pd  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
from util import get_data, plot_data  	
from scipy.optimize import minimize		



def plot_data(df, sd, ed, title="Normalized Portfolio Value vs SPY"):
   
    ax = df.plot(title=title, fontsize=12)
    ax.set_xlabel("Date")
    ax.set_ylabel("Normalized Price")
    ax.yaxis.grid(color='gray', linestyle='dashed')
    ax.xaxis.grid(color='gray', linestyle='dashed')
    ax.set_xlim(sd, ed)
    plt.savefig('figure.png')

def compute_daily_returns(df):
	daily_returns = df.copy()
	daily_returns[1:] = (df[1:] / df[:-1].values) - 1
	daily_returns.iloc[0, :] = 0 
	return daily_returns

def get_sr(normed, allocs): 
    pos_vals = normed * allocs
    port_val = pos_vals.sum(axis=1)
    daily_rets = compute_daily_returns(pd.DataFrame(port_val))
    daily_rets = daily_rets[1:]
    sr =  np.mean(daily_rets) / np.std(daily_rets)	
    return -sr


def optimize(normed, symbols):
    init_guess = np.ones(len(symbols)) * (1.0 / len(symbols))
    bounds = ((0.0, 1.0),) * len(symbols)
    weights = minimize(get_sr, init_guess,
                       args=(normed,), method='SLSQP',
                       options={'disp': False},
                       constraints=({'type': 'eq', 'fun': lambda inputs: 1.0 - np.sum(inputs)},
                                    ),
                       bounds=bounds)
    return weights.x

 		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
def optimize_portfolio(  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    sd=dt.datetime(2008, 1, 1),  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    ed=dt.datetime(2009, 1, 1),  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    syms=["GOOG", "AAPL", "GLD", "XOM"],  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    gen_plot=False,  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
):  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
	  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 	  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 	  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    dates = pd.date_range(sd, ed)  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    prices_all = get_data(syms, dates)  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    prices = prices_all[syms] 		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    prices_SPY = prices_all["SPY"] 
    number_days = 252

    normed = prices/prices.iloc[0]
    allocs = optimize(normed=normed, symbols=syms)

    pos_vals = normed * allocs
    port_val = pos_vals.sum(axis=1)
    daily_rets = compute_daily_returns(pd.DataFrame(port_val))
    daily_rets = daily_rets[1:]
    cr = (port_val[-1]/port_val[0] - 1) #cumulative return 
    adr = daily_rets.mean() #average daily return 	
    sddr = daily_rets.std() #standard deviation of daily return  
    sr = np.sqrt(number_days) * np.mean(daily_rets) / np.std(daily_rets)		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
  	 	 	 		 		 	 		 		 	 		  	 	 			  	  		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	   		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    port_val = (allocs * prices).sum(axis=1) 		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    if gen_plot:  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        df_temp = pd.concat(  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
            [port_val, prices_SPY], keys=["Portfolio", "SPY"], axis=1  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        )  	
        df = df_temp/df_temp.iloc[0]	  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        plot_data(df, sd, ed) 		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    return allocs, cr, adr, sddr, sr  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
def test_code():  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 	  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    start_date = dt.datetime(2009, 1, 1)  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    end_date = dt.datetime(2010, 1, 1)  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    symbols = ["GOOG", "AAPL", "GLD", "XOM", "IBM"]  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    allocations, cr, adr, sddr, sr = optimize_portfolio(  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        sd=start_date, ed=end_date, syms=symbols, gen_plot=True  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    )  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    # Print statistics  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    print(f"Start Date: {start_date}")  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    print(f"End Date: {end_date}")  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    print(f"Symbols: {symbols}")  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    print(f"Allocations:{allocations}")  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    print(f"Sharpe Ratio: {sr}")  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    print(f"Volatility (stdev of daily returns): {sddr}")  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    print(f"Average Daily Return: {adr}")  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    print(f"Cumulative Return: {cr}")  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
if __name__ == "__main__":  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    # This code WILL NOT be called by the auto grader  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    # Do not assume that it will be called  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    test_code()  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
