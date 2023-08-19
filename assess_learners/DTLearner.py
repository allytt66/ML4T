import numpy as np  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
class DTLearner(object):  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
	  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    def __init__(self, leaf_size, verbose=False):  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        self.leaf_size = leaf_size 
        self.verbose = verbose		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    def author(self):  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	   	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        return "yzhang3946" 	  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 	 
    def add_evidence(self, data_x, data_y): 	
        self.tree = self.build_tree(data_x, data_y)   

    def build_tree(self, data_x, data_y):  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 	  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        if self.is_terminal(data_x, data_y, self.leaf_size): 
            return np.array([[np.nan, np.mean(data_y), np.nan, np.nan]])
        # split_col, split_val, x_left, y_left, x_right, y_right = self.split_tree(data_x, data_y)
        
        split_col, split_val, mask = self.split_tree(data_x, data_y)

        if np.all(np.isclose(mask, mask[0])):
            return np.array([[np.nan, np.mean(data_y), np.nan, np.nan]])

        left_tree = self.build_tree(data_x[mask], data_y[mask])
        right_tree = self.build_tree(data_x[np.logical_not(mask)], data_y[np.logical_not(mask)])      
        
        root = np.array([[split_col, split_val, 1, left_tree.shape[0]+1]])
        return np.concatenate((root, left_tree, right_tree), axis=0)		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    def query(self, points): 

        predict_y = []
        for point in points: 
            position = 0 
            is_leaf = False
            while not is_leaf: 
                node = self.tree[position] 
                if np.isnan(node[0]): 
                    value_y= node[1]
                    is_leaf = True 
                else: 
                    if point[int(node[0])] <= node[1]: 
                        position = int(position + node[2])
                    else: 
                        position = int(position + node[3]) 
            predict_y.append(value_y) 	
        return np.array(predict_y) 	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
	  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
         		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	
    def split_tree(self, data_x, data_y): 
        corr_matrix = np.abs(np.corrcoef(data_x, data_y, rowvar=False))
        corr = corr_matrix[-1][:data_x.shape[1]]
        corr[np.isnan(corr)] = 0 
        i = np.argsort(corr)[-1]
        split_val = np.median(data_x[:,i])
        mask = data_x[:,i] <= split_val
        return i, split_val, mask

    
    def is_terminal(self, data_x, data_y, leaf_size): 
        return data_x.shape[0] <= leaf_size or len(set(data_y)) ==1  		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
if __name__ == "__main__":  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    print("the secret clue is 'zzyzx'")  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
