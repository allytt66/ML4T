import LinRegLearner as lrl
import BagLearner as bl 
import numpy as np 	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
class InsaneLearner(object):  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 	  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    def __init__(self, verbose=False):  
        self.learners =[bl.BagLearner(learner=lrl.LinRegLearner, kwargs={}, bags=20, boost=False)] * 20 		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    def author(self):  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        return "yzhang3946"  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    def add_evidence(self, data_x, data_y):  
        for learner in self.learners: 
            learner.add_evidence(data_x, data_y)		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	  	 	 		 		 	 		 		 	 		  	 	 			  	 
    def query(self, points): 
        out = []
        for learner in self.learners: 
            out.extend([learner.query(points)])
            return np.mean(out, axis=0) 		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 	  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	   	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
if __name__ == "__main__":  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    print("the secret clue is 'zzyzx'")  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
