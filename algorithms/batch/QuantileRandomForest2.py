import numpy as np
# from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
# from skgarden import RandomForestQuantileRegressor
import datetime as dat
import sys
sys.path.append('.../')
from configuration.config import * 


class RandomForestQuantileRegressor(RandomForestRegressor):
    def predict(self, X, quantile):
        if quantile is None:
            return super().predict(X)
        else:
            per_tree_pred = [pred.predict(X) for pred in self.estimators_]
            predictions = np.stack(per_tree_pred)
            predictions.sort(axis=0)
            return predictions[int(np.round(len(per_tree_pred) * quantile)), :]
            
            # result = np.quantile(predictions, quantile, axis=0).reshape(-1)
            # return result

class QRF():
    def __init__(self, alpha, input_train, output_train):
        self.alpha = alpha
        self.input_train = input_train
        self.output_train = output_train
        # self.output_max = np.max(output)
        
    def pre_learning(self):
        self.lower = []
        self.upper = []
        print('train1')
        print(len(self.input_train))
    
    def predict(self, input_test, num_split=num_split, num_estimator=num_estimator, max_depth=max_depth):
        self.output_train = self.output_train.reshape(-1)
        print('train2')
        print(len(self.input_train))
        if num_split > 0:
            # kf = KFold(n_splits=num_split, shuffle=True, random_state=0)
            kf = KFold(n_splits=num_split)
            
            for train_index, test_index in kf.split(self.input_train):
                rfqr = RandomForestQuantileRegressor(random_state=0, n_estimators=int(num_estimator), min_samples_leaf=1, min_samples_split=2, max_samples=0.25, max_depth=max_depth)
        
                x_train, y_train, x_test, y_test = (self.input_train[train_index], self.output_train[train_index], self.input_train[test_index], self.output_train[test_index]) 
               
                rfqr.fit(x_train, y_train)
                print('train')
                print(len(x_train))
                print('test')
                print(len(x_test))
                self.lower = np.concatenate((self.lower, rfqr.predict(x_test, quantile=self.alpha[0]))) 
                self.upper = np.concatenate((self.upper, rfqr.predict(x_test, quantile=self.alpha[1])))
                print('up')
                print(len(self.upper))

        else:            
            # rfqr.fit(input_sort, output_test)
            rfqr = RandomForestQuantileRegressor(random_state=0, n_estimators=int(num_estimator), min_samples_leaf=1, min_samples_split=2, max_samples=0.01, max_depth=max_depth)
    
            rfqr.fit(self.input_train, self.output_train)
            
            self.lower = rfqr.predict(self.input_train, quantile=self.alpha[0])
            self.upper = rfqr.predict(self.input_train, quantile=self.alpha[1])
            print('up')
            print(self.upper)
        
        result = np.vstack((self.lower.T, self.upper.T)).reshape(2, -1)
        print('result')
        print(np.size(result))
                
        return result