""""""  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
"""  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
Template for implementing StrategyLearner  (c) 2016 Tucker Balch  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
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
  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
Student Name: Yueting Zhang 		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
GT User ID: yzhang3946		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
GT ID: 903862041		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
"""	  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
import datetime as dt  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
import random as rand  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 	  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
import pandas as pd  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
import util as ut  	
import QLearner as ql 
import indicators as ind
from marketsimcode import compute_portvals
import numpy as np 


class StrategyLearner(): 

    def __init__(
            self,
            verbose, 
            impact, 
            commission
    ): 

        self.verbose = verbose 
        self.impact = impact 
        self.commission = commission
        self.learner = ql.QLearner(
            num_states=27, 
            num_actions=3, 
            alpha=0.2, 
            gamma = 0.9, 
            rar = 0.5,
            radr = 0.99, 
            dyna = 50,
            verbose= False
        )

        rand.seed(903862041)

		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    def add_evidence(  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        self,  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        symbol="JPM",  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        sd=dt.datetime(2008, 1, 1),  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        ed=dt.datetime(2009, 12, 31),  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        sv=100000,  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    ):  		  

        train_state = self.get_price_state(sd, ed, symbol)
        epochs = 1

        for epoch in range(1, epochs + 1): 

            i = 0
            while i < len(train_state) -1: 

                state = int(train_state.iloc[i]['state'])
                cur_price = train_state.iloc[i]['price']
                
                if i == 0: 
                    action = self.learner.querysetstate(state)
                else: 
                    action = self.learner.query(state, reward)

                holding = (action - 1) * 1000
                next_price =  train_state.iloc[i + 1]['price']
                reward = (next_price - cur_price) * holding 

                i += 1 

        # return self.learner.q

                                                                                                                                        
    def testPolicy(  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        self,  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        symbol="JPM",  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        sd=dt.datetime(2010, 1, 1),  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        ed=dt.datetime(2011, 12, 31),  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        sv=10000,  	  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    ):  
        
        test_state = self.get_price_state(sd, ed, symbol)
        action_list = []

        i = 0 
        while i < len(test_state): 
            state = int(test_state.iloc[i]['state'])
            action = self.learner.test_action(state) 
            action_list.append(action)
            i += 1 

        df_trade_strategy = self.generate_trade(action_list, test_state, symbol)


        return df_trade_strategy

    def generate_trade(self, actions_list, state, symbol): 

        actions_df = pd.DataFrame(actions_list).set_index(state.index)
        actions_df = actions_df - 1
        trades = (actions_df - actions_df.shift(1)).fillna(actions_df.iloc[0].values[0])
        trades = trades * 1000 

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

                                                                                                                                    

    def get_price_state(self, sd, ed, symbol): 
        """
        for the period, get prices and converted stated in a dataframe 
        """

        syms = [symbol]  
        dates = pd.date_range(sd, ed) 

        prices= ut.get_data(syms, dates)[syms]
        vol = ut.get_data(syms, dates, colname='Volume')[syms]

        price_thres, vol_thres = self.get_historical_range(sd, ed, 'JPM')

        price_data = pd.DataFrame(prices[symbol]).rename(columns={symbol: 'price'})
        prices_state = pd.DataFrame(pd.cut(prices[symbol], bins=price_thres, labels=[1,2,3]))
        vol_state = pd.DataFrame(pd.cut(vol[symbol], bins=vol_thres, labels=[1,2,3]))

        prices_state = prices_state.rename(columns={symbol: 'price'})
        vol_state = vol_state.rename(columns={symbol: 'vol'})

        history = dt.timedelta(100)
        full_sd = sd - history
        df_price_full = ut.get_data([symbol], pd.date_range(full_sd, ed))
        df_price_full = df_price_full[syms]
        lookback = 60 

        bbp = ind.bbp(sd, ed, lookback, df_price_full)
        bbp_thres = [-np.inf,0,1,np.inf]
        bbp_state = pd.DataFrame(pd.cut(bbp['bbp'], bins=bbp_thres, labels=[1,2,3])) 

        observations = pd.concat([prices_state, vol_state, bbp_state], axis=1)
        state = pd.DataFrame(observations.apply(self.convert_state, axis=1))
        state_price= pd.concat([state, price_data], axis=1).rename(columns={0: "state"})

        return state_price



    def get_historical_range(self, hsd, hed, symbol):

        #use historial price and vol ranges to discretize states 
        hsd=dt.datetime(2003, 1, 1)		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        hed=dt.datetime(2007, 12, 31)
        
        syms = [symbol]  	
        hdates = pd.date_range(hsd, hed) 
        hprices = ut.get_data(syms, hdates)[syms]
        hvol = ut.get_data(syms, hdates, colname='Volume')[syms]
        price_range = hprices.describe()
        vol_range = hvol.describe()

        price_thres = [-np.inf, 
                        price_range[symbol]['25%'], 
                        price_range[symbol]['75%'],  
                        np.inf ]
        
        vol_thres =  [-np.inf, 
                    vol_range[symbol]['25%'], 
                    vol_range[symbol]['75%'],  
                    np.inf ]
        
        return price_thres, vol_thres		  	  		

    def convert_state(self, row): 
        return (row['price'] - 1) * 9 + (row['vol'] -1) * 3 + (row['bbp'] - 1)	  


                                                                                                                                                                                                                                                                                                    
  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
if __name__ == "__main__":  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    print("One does not simply think up a strategy")  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
