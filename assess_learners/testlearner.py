""""""  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
"""  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
Test a learner.  (c) 2015 Tucker Balch  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
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
"""  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
import math  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
import sys  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
import numpy as np  
from LinRegLearner import LinRegLearner
from DTLearner import DTLearner
from RTLearner import RTLearner	
from BagLearner import BagLearner
from InsaneLearner import InsaneLearner 
import matplotlib.pyplot as plt 
import timeit
import time
import sys 


def research_overfit(train_x, train_y, test_x, test_y, learner, plot_range=100, metrics="rmse"): 
    in_sample_rmses = []
    out_sample_rmses = []

    for leaf_size in range(1, plot_range): 
        
        if learner is DTLearner or learner is RTLearner:
            learner_instance = learner(leaf_size)
        if learner is BagLearner:
            learner_instance = learner(DTLearner,kwargs={"leaf_size": leaf_size}, bags=30, boost=False)
        
        learner_instance.add_evidence(train_x, train_y)

        if metrics == "rmse": 
            in_sample_rmse = np.linalg.norm(learner_instance.query(train_x) - train_y) / np.sqrt(train_y.shape[0])
            out_sample_rmse = np.linalg.norm(learner_instance.query(test_x) - test_y) / np.sqrt(test_y.shape[0])
        else: 
            in_sample_rmse = np.mean(np.abs((np.asarray(learner_instance.query(train_x) - np.asarray(train_y)))))
            out_sample_rmse = np.mean(np.abs((np.asarray(learner_instance.query(test_x) - np.asarray(test_y)))))

        in_sample_rmses.extend([in_sample_rmse])
        out_sample_rmses.extend([out_sample_rmse])

    return in_sample_rmses, out_sample_rmses

def research_runtime(train_x, train_y): 

    runtime_dts = []
    runtime_rts = []

    for sample_size in range(10, train_x.shape[0], 10): 
        sample_x = train_x[:sample_size]
        sample_y = train_y[:sample_size]

        learner = DTLearner(leaf_size=1)
        start = time.time()
        learner.add_evidence(train_x, train_y)
        end = time.time()
        runtime_dts.extend([end-start])

        learner = RTLearner(leaf_size=1)
        start = time.time()
        learner.add_evidence(train_x, train_y)
        end = time.time()
        runtime_rts.extend([end-start])

    return runtime_dts, runtime_rts



def gen_plot(in_sample_rmses, out_sample_rmses, title, data_x="in_sample", data_y="out_sample", x_label ="leaf size", y_label="rmse",  x_axis = range(1, 100)): 
    plt.clf()
    plt.plot(x_axis, in_sample_rmses, label = data_x)
    plt.plot(x_axis, out_sample_rmses, label = data_y)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    # plt.xticks(np.arange(5, 100, step=5))
    plt.grid()
    plt.legend()
    plt.savefig(f'images/{title}.png')		  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
if __name__ == "__main__":  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    if len(sys.argv) != 2:  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        print("Usage: python testlearner.py <filename>")  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        sys.exit(1)  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    inf = open(sys.argv[1])  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    data = np.array(  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        [list(map(str, s.strip().split(","))) for s in inf.readlines()]  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    )  		

    if sys.argv[1] =="Data/Istanbul.csv": 
        data = data[1:, 1:]

    data = data.astype('float')
			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    train_rows = int(0.6 * data.shape[0])  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    test_rows = data.shape[0] - train_rows  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  			  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    train_x = data[:train_rows, 0:-1]  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    train_y = data[:train_rows, -1]  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    test_x = data[train_rows:, 0:-1]  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    test_y = data[train_rows:, -1]  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
 		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    learner = LinRegLearner(verbose=True)  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    print(learner.author())  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 	  	  
    start = timeit.default_timer()
    experiment_1_ins, experiment_1_outs = research_overfit(train_x, train_y, test_x, test_y, DTLearner)	
    stop = timeit.default_timer()
    print('DT Learner', stop - start)
    
    gen_plot(experiment_1_ins, experiment_1_outs, title = "leaf size vs overfit for DTlearner")  


    start = timeit.default_timer()
    experiment_2_ins, experiment_2_outs = research_overfit(train_x, train_y, test_x, test_y, BagLearner)
    stop = timeit.default_timer()
    print('Bag Learner', stop - start)

    gen_plot(experiment_2_ins, experiment_2_outs, title= "leaf size vs overfit with bagging")

    start = timeit.default_timer()
    experiment_3_ins_rt_mae, experiment_3_outs_rt_mae = research_overfit(train_x, train_y, test_x, test_y, RTLearner, metrics="mae")
    stop = timeit.default_timer()
    print('RT Learner', stop - start)
    
    experiment_3_ins_dt_mae, experiment_3_outs_dt_mae = research_overfit(train_x, train_y, test_x, test_y, DTLearner, metrics="mae")	


    gen_plot(experiment_3_ins_dt_mae, experiment_3_ins_rt_mae, title= "in sample leaf size vs overfit comparison", data_x="DTLearner", data_y="RTLearner", y_label="mae") 	
    gen_plot(experiment_3_outs_dt_mae, experiment_3_outs_rt_mae, title= "out sample leaf size vs overfit comparison", data_x="DTLearner", data_y="RTLearner", y_label="mae") 

    runtime_dt, runtime_rt = research_runtime(train_x, train_y)
    gen_plot(runtime_dt, runtime_rt, title= "dt vs rt run time comparison", data_x="DTLearner", data_y="RTLearner", x_label="training size", y_label="time", x_axis=range(10, train_x.shape[0], 10))

    





