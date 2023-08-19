import math  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
import os  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
import random  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
import string  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
import sys  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
import time  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
import traceback as tb  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
from collections import namedtuple  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
import numpy as np  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
import pandas as pd  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
import util  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 


test_x, test_y, train_x, train_y = None, None, None, None  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
permutation = None  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
author = None  		  
datafile = "Istanbul.csv"  		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
with util.get_learner_data_file(datafile) as f:  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    alldata = np.genfromtxt(f, delimiter=",")  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    # Skip the date column and header row if we're working on Istanbul data  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    if datafile == "Istanbul.csv":  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        alldata = alldata[1:, 1:]  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    datasize = alldata.shape[0]  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    cutoff = int(datasize * 0.6)  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    permutation = np.random.permutation(alldata.shape[0])  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    col_permutation = np.random.permutation(alldata.shape[1] - 1)  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    train_data = alldata[permutation[:cutoff], :]  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    # train_x = train_data[:,:-1]  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    train_x = train_data[:, col_permutation]  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    train_y = train_data[:, -1]  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    test_data = alldata[permutation[cutoff:], :]  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    # test_x = test_data[:,:-1]  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    test_x = test_data[:, col_permutation]  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    test_y = test_data[:, -1]  


def add_evidence(data_x, data_y, leaf_size):   

    if is_terminal(data_x, data_y, leaf_size): 
        return np.array([[999, np.mean(data_y), np.nan, np.nan]])
    
    split_col, split_val, x_left, y_left, x_right, y_right = split_tree(data_x, data_y)
    if is_terminal(x_left, y_left, leaf_size): 
        return np.array([[999, np.mean(y_left), np.nan, np.nan]])
    else: 
        left_tree = add_evidence(x_left, y_left, leaf_size)
    
    if is_terminal(x_right, y_right, leaf_size):
        return np.array([[999, np.mean(y_right), np.nan, np.nan]])
    else:    
        right_tree = add_evidence(x_right, y_right, leaf_size)      
    
    root = np.array([[split_col, split_val, 1, left_tree.shape[0]+1]])
    return np.concatenate((root, left_tree, right_tree), axis=0)
    

def split_tree(data_x, data_y): 
    corr_matrix = np.abs(np.corrcoef(data_x, data_y, rowvar=False))
    corr = corr_matrix[-1][:data_x.shape[1]]
    i = np.argsort(corr)[-1]
    split_val = np.median(data_x[:,i])
    x_left = data_x[data_x[:,i] <= split_val]
    y_left = data_y[data_x[:,i] <= split_val]
    x_right = data_x[data_x[:,i] > split_val]
    y_right = data_y[data_x[:,i] > split_val]
    # handle edge case, when there is no split, split half half 
    if x_left.shape[0] == data_x.shape[0]: 
        split = int(np.floor(train_x.shape[0]/2))
        train_x[:split].shape
        train_x[split:].shape
        x_left = data_x[:split]
        y_left = data_y[:split]
        x_right = data_x[split:]
        y_right = data_y[split:]
    
    return i, split_val, x_left, y_left, x_right, y_right

def is_terminal(data_x, data_y, leaf_size): 
    return data_x.shape[0] <= leaf_size or len(set(data_y)) ==1


print(add_evidence(test_x, test_y, 1))