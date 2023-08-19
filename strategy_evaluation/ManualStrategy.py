import util as ut
from marketsimcode import compute_portvals
import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 
import indicators as ind 



class ManualStrategy(): 

    def __init___(self): 
        pass 

    def testPolicy(self, symbol, sd, ed, sv):
        
        """
        manual_trades = manual.testPolicy([symbol], sd=sd, ed=ed, sv=sv)
        return a dataframe of trades 
        """

        prices = ut.get_data([symbol], pd.date_range(sd, ed))[[symbol]]

        volume = ut.get_data([symbol],  pd.date_range(sd, ed), colname='Volume')[[symbol]]
        volume = volume.rename(columns={symbol: 'vol'})


        history = dt.timedelta(100)
        full_sd = sd - history
        df_price_full = ut.get_data([symbol], pd.date_range(full_sd, ed))
        df_price_full = df_price_full[[symbol]]
        lookback = 60 
        

        bbp = ind.bbp(sd, ed,symbol, lookback, df_price_full)
        sma = ind.price_sma(sd, ed, symbol, lookback, df_price_full)
        rsi = ind.rsi(sd, ed, symbol, lookback, df_price_full)
        osi = ind.stochastic_osi(sd, ed, symbol, lookback, df_price_full)
        chaikin = ind.chaikin(sd, ed, symbol, lookback, df_price_full)

        observations = pd.concat([prices, bbp, sma, rsi, osi, chaikin, volume], axis=1)

        period = 2

        bbp_thres = [-np.inf,0,1,np.inf]
        bbp_signal = pd.DataFrame(pd.cut(observations['bbp'], bins=bbp_thres, labels=[-1,0,1])) 
        bbp_signal['rolling']=bbp_signal.rolling(period).sum()
        sell_sig_bbp = bbp_signal[(bbp_signal['rolling'] == period-1) & (bbp_signal['bbp'] == 0)]
        buy_sig_bbp = bbp_signal[(bbp_signal['rolling'] == -(period-1)) & (bbp_signal['bbp'] == 0)]


        rsi_thres = [-np.inf, 30, 70, np.inf]
        rsi_signal = pd.DataFrame(pd.cut(observations['rsi'], bins=rsi_thres, labels=[-1,0,1])) 
        rsi_signal['rolling']=rsi_signal.rolling(period).sum()
        sell_sig_rsi= rsi_signal[(rsi_signal['rolling'] == period-1) & (rsi_signal['rsi'] == 0)]
        buy_sig_rsi = rsi_signal[(rsi_signal['rolling'] == -(period-1)) & (rsi_signal['rsi'] == 0)]

        sma_thres = [-np.inf, 0.9, 1.1, np.inf]
        sma_signal = pd.DataFrame(pd.cut(observations['sma'], bins=sma_thres, labels=[-1,0,1])) 
        sma_signal['rolling']=sma_signal.rolling(period).sum()
        sell_sig_sma= sma_signal[(sma_signal['rolling'] == period-1) & (sma_signal['sma'] == 0)]
        buy_sig_sma = sma_signal[(sma_signal['rolling'] == -(period-1)) & (sma_signal['sma'] == 0)]

        trading_signals = pd.concat([buy_sig_bbp, buy_sig_rsi, buy_sig_sma, sell_sig_bbp, sell_sig_rsi, sell_sig_sma]).sort_index()

        df_trade_manual = self.generate_trade(trading_signals, symbol)

        return df_trade_manual


    def generate_trade(self, trading_signals, symbol):

        df_trade_manual = pd.DataFrame(columns=['Date', 'Symbol', 'Order','Shares'])

        for i in range(len(trading_signals)): 
            if i==0: 
                if trading_signals.iloc[i]['rolling'] == -1: 
                    order = "BUY"
                else: 
                    order = "SELL"
                data = {'Date': trading_signals.index[i], 
                        'Symbol': symbol, 
                        'Order':order, 
                        'Shares':1000}
                df_trade_manual = df_trade_manual.append(data, ignore_index=True)
            
            else: 
                if trading_signals.iloc[i - 1]['rolling'] != trading_signals.iloc[i]['rolling']: 
                    if trading_signals.iloc[i]['rolling'] == -1: 
                        order = "BUY"
                    else: 
                        order = "SELL"
                    data = {'Date': trading_signals.index[i], 
                            'Symbol': symbol, 
                            'Order':order, 
                            'Shares':2000}
                    df_trade_manual = df_trade_manual.append(data, ignore_index=True)
        return df_trade_manual


    def get_stats(self, portvals):
        cr = portvals[-1] / portvals[0] - 1
        dr = (portvals / portvals.shift(1) - 1).iloc[1:]
        sddr = dr.std()
        adr = dr.mean()
        return cr, sddr, adr


    def plot_performance(self, sd, ed, symbol, fig_name): 
        prices_insample =  ut.get_data([symbol], pd.date_range(sd, ed))[[symbol]]
   
        manual_trades = self.testPolicy(symbol=symbol, sd=sd, ed=ed, sv=sv)
        manual_insample = compute_portvals(manual_trades, start_val=sv, commission=0, impact = 0.000, start=sd, end=ed)
        manual_insample = manual_insample/manual_insample[0]

        bench_trades = pd.DataFrame(columns=['Date', 'Symbol', 'Order','Shares'])
        data = {'Date': prices_insample.index[0], 
            'Symbol': symbol, 
            'Order':'BUY', 
            'Shares':1000}
        bench_trades = bench_trades.append(data, ignore_index=True)
        bench_insample = compute_portvals(bench_trades, start_val = sv, commission=0, impact=0.000, start=sd, end=ed)
        bench_insample = bench_insample/bench_insample[0]

        fig, ax = plt.subplots()
        ax.plot(manual_insample, color="red", label="manual")
        ax.plot(bench_insample, color="purple", label="benchmark")
        ax.set_ylabel('normalized return')
        ax.grid()
        ax.legend()
        ax.set_title('strategy comparison')


        for i in manual_trades.index:
            if manual_trades.iloc[i]['Order'] == 'BUY':
                ax.axvline(x = manual_trades.iloc[i]['Date'], color = 'blue', linestyle='--', label='Vertical Line')
            else: 
                ax.axvline(x = manual_trades.iloc[i]['Date'], color = 'black', linestyle='--', label='Vertical Line')

        fig.set_figwidth(14)

        fig.savefig(fig_name)

        cr_m, sddr_m, adr_m = self.get_stats(manual_insample)
        cr_b, sddr_b, adr_b = self.get_stats(bench_insample)

        print('manual')
        print(f"cumulative return {cr_m}")
        print(f"sddr {sddr_m}")
        print(f"adr {adr_m}")

        print('bench')
        print(f"cumulative return {cr_b}")
        print(f"sddr {sddr_b}")
        print(f"adr {adr_b}")

 

    def author():
        return "yzhang3946"
    
    
if __name__ == '__main__': 
    ms = ManualStrategy()
    symbol = 'JPM'
    sd=dt.datetime(2008, 1, 1)		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    ed=dt.datetime(2009, 12, 31)  

    tsd=dt.datetime(2010, 1, 1)		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    ted=dt.datetime(2011, 12, 31)  

    sv = 100000

    ms.plot_performance(sd, ed, 'JPM', 'insample.png')
    ms.plot_performance(tsd, ted, 'JPM', 'outsample.png')
    