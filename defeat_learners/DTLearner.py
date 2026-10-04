""""""  		  	   		 		  		  		  		    	 		 		   		 		  
"""  		  	   		 		  		  		  		    	 		 		   		 		  
A simple wrapper for linear regression.  (c) 2015 Tucker Balch  		  	   		 		  		  		  		    	 		 		   		 		  
Note, this is NOT a correct DTLearner; Replace with your own implementation.  		  	   		 		  		  		  		    	 		 		   		 		  
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
  		  	   		 		  		  		  		    	 		 		   		 		  
  		  	   		 		  		  		  		    	 		 		   		 		  
class DTLearner(object):  		  	   		 		  		  		  		    	 		 		   		 		  
    """  		  	   		 		  		  		  		    	 		 		   		 		  
    This is a decision tree learner object that is implemented incorrectly. You should replace this DTLearner with  		  	   		 		  		  		  		    	 		 		   		 		  
    your own correct DTLearner from Project 3.  		  	   		 		  		  		  		    	 		 		   		 		  
  		  	   		 		  		  		  		    	 		 		   		 		  
    :param leaf_size: The maximum number of samples to be aggregated at a leaf, defaults to 1.  		  	   		 		  		  		  		    	 		 		   		 		  
    :type leaf_size: int  		  	   		 		  		  		  		    	 		 		   		 		  
    :param verbose: If “verbose” is True, your code can print out information for debugging.  		  	   		 		  		  		  		    	 		 		   		 		  
        If verbose = False your code should not generate ANY output. When we test your code, verbose will be False.  		  	   		 		  		  		  		    	 		 		   		 		  
    :type verbose: bool  		  	   		 		  		  		  		    	 		 		   		 		  
    """  		  	   		 		  		  		  		    	 		 		   		 		  
  		  	   		 		  		  		  		    	 		 		   		 		  
    def __init__(self, leaf_size=1, verbose=False):  		  	   		 		  		  		  		    	 		 		   		 		  
        """  		  	   		 		  		  		  		    	 		 		   		 		  
        Constructor method  		  	   		 		  		  		  		    	 		 		   		 		  
        """  		  	   		 		  		  		  		    	 		 		   		 		  
        self.leaf_size = leaf_size
        self.verbose = verbose
        self.tree = None
  		  	   		 		  		  		  		    	 		 		   		 		  
    def author(self):  		  	   		 		  		  		  		    	 		 		   		 		  
        """  		  	   		 		  		  		  		    	 		 		   		 		  
        :return: The GT username of the student  		  	   		 		  		  		  		    	 		 		   		 		  
        :rtype: str  		  	   		 		  		  		  		    	 		 		   		 		  
        """  		  	   		 		  		  		  		    	 		 		   		 		  
        return "rwong80"

    def study_group(self):
        """Return the GT usernames of the study group members."""
        return "rwong80"
  		  	   		 		  		  		  		    	 		 		   		 		  
    def add_evidence(self, data_x, data_y):  		  	   		 		  		  		  		    	 		 		   		 		  
        """  		  	   		 		  		  		  		    	 		 		   		 		  
        Add training data to learner  		  	   		 		  		  		  		    	 		 		   		 		  
  		  	   		 		  		  		  		    	 		 		   		 		  
        :param data_x: A set of feature values used to train the learner  		  	   		 		  		  		  		    	 		 		   		 		  
        :type data_x: numpy.ndarray  		  	   		 		  		  		  		    	 		 		   		 		  
        :param data_y: The value we are attempting to predict given the X data  		  	   		 		  		  		  		    	 		 		   		 		  
        :type data_y: numpy.ndarray  		  	   		 		  		  		  		    	 		 		   		 		  
        """  		  	   		 		  		  		  		    	 		 		   		 		  
  		  	   		 		  		  		  		    	 		 		   		 		  
        self.tree = self._build_tree(data_x, data_y)

    def _build_tree(self, data_x, data_y):
        """Recursively build a tree stored as feature, split, and row offsets."""
        if data_x.shape[0] <= self.leaf_size or np.all(data_y == data_y[0]):
            return np.array([[-1.0, np.mean(data_y), np.nan, np.nan]])

        correlations = np.zeros(data_x.shape[1])
        for feature in range(data_x.shape[1]):
            if np.std(data_x[:, feature]) > 0.0:
                correlations[feature] = abs(
                    np.corrcoef(data_x[:, feature], data_y)[0, 1]
                )
        correlations = np.nan_to_num(correlations)
        split_feature = int(np.argmax(correlations))
        split_value = np.median(data_x[:, split_feature])
        left_rows = data_x[:, split_feature] <= split_value

        if np.all(left_rows) or not np.any(left_rows):
            return np.array([[-1.0, np.mean(data_y), np.nan, np.nan]])

        left_tree = self._build_tree(data_x[left_rows], data_y[left_rows])
        right_tree = self._build_tree(data_x[~left_rows], data_y[~left_rows])
        root = np.array(
            [[split_feature, split_value, 1, left_tree.shape[0] + 1]]
        )
        return np.vstack((root, left_tree, right_tree))
  		  	   		 		  		  		  		    	 		 		   		 		  
    def query(self, points):  		  	   		 		  		  		  		    	 		 		   		 		  
        """  		  	   		 		  		  		  		    	 		 		   		 		  
        Estimate a set of test points given the model we built.  		  	   		 		  		  		  		    	 		 		   		 		  
  		  	   		 		  		  		  		    	 		 		   		 		  
        :param points: A numpy array with each row corresponding to a specific query.  		  	   		 		  		  		  		    	 		 		   		 		  
        :type points: numpy.ndarray  		  	   		 		  		  		  		    	 		 		   		 		  
        :return: The predicted result of the input data according to the trained model  		  	   		 		  		  		  		    	 		 		   		 		  
        :rtype: numpy.ndarray  		  	   		 		  		  		  		    	 		 		   		 		  
        """  		  	   		 		  		  		  		    	 		 		   		 		  
        predictions = np.empty(points.shape[0])
        for point_index, point in enumerate(points):
            row = 0
            while int(self.tree[row, 0]) != -1:
                feature = int(self.tree[row, 0])
                offset_column = 2 if point[feature] <= self.tree[row, 1] else 3
                row += int(self.tree[row, offset_column])
            predictions[point_index] = self.tree[row, 1]
        return predictions
  		  	   		 		  		  		  		    	 		 		   		 		  
  		  	   		 		  		  		  		    	 		 		   		 		  
if __name__ == "__main__":  		  	   		 		  		  		  		    	 		 		   		 		  
    print("the secret clue is 'zzyzx'")  		  	   		 		  		  		  		    	 		 		   		 		  
