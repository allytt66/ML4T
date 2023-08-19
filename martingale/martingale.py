""""""  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
"""Assess a betting strategy.  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
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
GT User ID: tb34 (replace with your User ID)  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
GT ID: 900897987 (replace with your GT ID)  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
"""  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
import numpy as np  
import pandas as pd		
import matplotlib.pyplot as plt  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
def author():  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    """  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    :return: The GT username of the student  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    :rtype: str  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    """  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    return "yzhang3946"  # replace tb34 with your Georgia Tech username.  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
def gtid():  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    """  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    :return: The GT ID of the student  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    :rtype: int  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    """  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    return 903862041  # replace with your GT ID number  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
def get_spin_result(win_prob):  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    """  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    Given a win probability between 0 and 1, the function returns whether the probability will result in a win.  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    :param win_prob: The probability of winning  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    :type win_prob: float  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    :return: The result of the spin.  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    :rtype: bool  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    """  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    result = False  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    if np.random.random() <= win_prob:  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        result = True  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    return result  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	   		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	

def bet_strategy(): 
    episode_winnings = 0 
    records = np.zeros(1000)
    num_spins = 0
    while episode_winnings < 80: 
        won = False
        bet_amount = 1 
        while not won: 
            won = get_spin_result(18/38)
            if won == True: 
                episode_winnings = episode_winnings + bet_amount
                records[num_spins]= episode_winnings
                num_spins += 1
            else: 
                episode_winnings = episode_winnings - bet_amount
                bet_amount = bet_amount * 2 
                records[num_spins]= episode_winnings
                num_spins += 1
    records[num_spins:]= records[num_spins - 1]
    return records 


def bet_strategy_realistic(): 
    episode_winnings = 0 
    records = np.zeros(1000)
    num_spins = 0
    bankroll = 256 
    while episode_winnings < 80 and bankroll > 0 and num_spins <1000 :  
        won = False
        bet_amount = 1 
        while not won and bankroll > 0 and num_spins <1000 : 
            won = get_spin_result(18/38)
            if won == True: 
                episode_winnings = episode_winnings + bet_amount
                bankroll = bankroll + bet_amount 
            else: 
                episode_winnings = episode_winnings - bet_amount
                bankroll = bankroll - bet_amount 
                bet_amount = min(bet_amount * 2, bankroll)
            records[num_spins]= episode_winnings
            num_spins += 1
           
    records[num_spins:]= records[num_spins - 1]
    return records 


def record_episode(num_episode, realistic=True):
    num_spins = 1000 
    outcome = np.zeros((num_episode, num_spins))
    for i in range(0, num_episode): 
        if realistic: 
            outcome[i, :] = bet_strategy_realistic()
        else: 
            outcome[i, :] = bet_strategy()
    return outcome

def summarize(df): 
    df_summary = pd.DataFrame(df.mean(), columns=['mean'])
    df_median = pd.DataFrame(df.median(), columns=['median'])
    df_std = pd.DataFrame(df.std(), columns=['std'])
    df_summary = df_summary.join(df_median).join(df_std)
    df_summary['mean -std'] = df_summary['mean']- df_summary['std']
    df_summary['mean +std'] = df_summary['mean']+ df_summary['std']
    df_summary['median -std'] = df_summary['median']- df_summary['std']
    df_summary['median +std'] = df_summary['median']+ df_summary['std']
    return df_summary


def plot_data(df, title="Figure 1"): 
    ax = df.plot(title=title, fontsize=12, label='winnig')
    ax.set_xlabel("Spin")  
    ax.set_xlim([0, 300])  
    ax.set_ylim([-256, 100]) 
    ax.legend()
    plt.ylabel("Winning")
    plt.savefig('{}.png'.format(title))


def test_code():  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    """  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    Method to test your code  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    """  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    win_prob = 18/38 	

    experiment_1 = pd.DataFrame(record_episode(10, False))
    plot_data(experiment_1.loc[:,:299].transpose(), 'Figure 1')

    experiment_1 = pd.DataFrame(record_episode(1000, False))
    experiment_1 = summarize(experiment_1)
    plot_data(experiment_1[['mean', 'mean -std', 'mean +std']].loc[:299,:], 'Figure 2')

    plot_data(experiment_1[['median', 'median -std', 'median +std']].loc[:299,:], 'Figure 3')

    experiment_2 = pd.DataFrame(record_episode(1000, True))
    experiment_2 = summarize(experiment_2)

    plot_data(experiment_2[['mean', 'mean -std', 'mean +std']].loc[:299,:], 'Figure 4')
    plot_data(experiment_2[['median', 'median -std', 'median +std']].loc[:299,:], 'Figure 5')

 
  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
if __name__ == "__main__":  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    test_code()  	
   	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
