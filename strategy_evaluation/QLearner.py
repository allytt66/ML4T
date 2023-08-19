""""""
"""  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
Template for implementing QLearner  (c) 2015 Tucker Balch  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
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

import random as rand
import numpy as np
from collections import namedtuple

Experience = namedtuple("Experience", ["s", "a", "s_prime", "r"])

class QLearner(object):

    def __init__(
        self,
        num_states=100, # 10 x 10 grid states
        num_actions=4, # 4 direction moves 
        alpha=0.2, # The learning rate used in the update rule.
        gamma=0.9, #The discount rate used in the update rule.
        rar=0.5, # Random action rate: the probability of selecting a random action at each step. 
        radr=0.99, # Random action decay rate, after each update, rar = rar * radr.
        dyna=0, # The number of dyna updates for each regular update.
        verbose=False,
    ):
  
        self.num_states = num_states
        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma
        self.rar = rar 
        self.radr = radr 
        self.dyna = dyna 
        self.verbose = verbose
        
        self.s = 0
        self.a = 0
        self.q = np.zeros((num_states, num_actions))

        if dyna: 
            self.history = [] # list of namedtuple( state, action, s_prime, r)

    def querysetstate(self, s):

        if rand.uniform(0.0, 1.0) < self.rar:  
            action = rand.randint(0, self.num_actions - 1)
        else: 
            action = np.argmax(self.q[s])
        
        self.rar = self.rar * self.radr
        self.s = s
        self.a = action # keep track of those for q table update 

        if self.verbose:
            print(f"s = {s}, a = {action}")

        return action

    def query(self, s_prime, r):

        self.q[self.s, self.a] = (1 - self.alpha) * self.q[self.s, self.a] + self.alpha * (r + self.gamma * np.max(self.q[s_prime]))

        if self.dyna: 
            experience = Experience(self.s, self.a, s_prime, r)
            self.history.extend([experience])
  
            for _ in range(self.dyna): 
                ex = rand.choice(self.history)
                self.q[ex.s, ex.a] = (1 - self.alpha) * self.q[ex.s,ex.a] + self.alpha * (ex.r + self.gamma * np.max(self.q[ex.s_prime]))

        if rand.uniform(0.0, 1.0) < self.rar:  
            action = rand.randint(0, self.num_actions - 1)
        else: 
            action = np.argmax(self.q[s_prime])
        
        self.rar = self.rar * self.radr # decay it 
        self.s = s_prime
        self.a = action 

        if self.verbose:
            print(f"s = {s_prime}, a = {action}, r={r}")
        return action
    
    def test_action(self, s): 
        action = np.argmax(self.q[s])

        if self.verbose:
            print(f"s = {s}, a = {action}")
        return action 

    
    def author(self):
        return "yzhang3946"


if __name__ == "__main__":
    print("Remember Q from Star Trek? Well, this isn't him")
