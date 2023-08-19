""""""  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
"""  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
template for generating data to fool learners (c) 2016 Tucker Balch  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
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
  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
import math  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
import numpy as np  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
# this function should return a dataset (X and Y) that will work  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
# better for linear regression than decision trees  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
def best_4_lin_reg(seed=1489683273):  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	   	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    np.random.seed(seed)  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    x = np.random.random(size=(100, 6))		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    y = x[:,0] * 1 + x[:,1] * 2 + x[:,2] * 3 + x[:,3] * 4 + x[:, 4] * 5 + x[:, 5] * 6 + 100  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 	  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    return x, y  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
def best_4_dt(seed=1489683273):  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 	  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    np.random.seed(seed)  	
    zeros = np.zeros((100,))
    ones = np.ones((100,))	  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    x = np.random.random(size = (100, 3))		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    y = np.where(x[:,0] > 0.5, ones, zeros)	  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    return x, y  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
def author():  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    return "yzhang3946"   # Change this to your user ID  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
if __name__ == "__main__":  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    print("they call me Tim.")  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
