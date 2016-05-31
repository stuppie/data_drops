
"""
https://www.topcoder.com/challenge-details/30054204/?type=develop&nocache=true



"""
import sys
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import RidgeClassifierCV
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.externals import joblib
from sklearn import cross_validation
from sklearn.cross_validation import KFold

import pandas as pd
import numpy as np


def predict(test_data, out_file):
    df=pd.read_excel(test_data)
    df = df.fillna('')
    x1=list(df["#water_tech"].str.lower())
    x2=list(df["#water_source"].str.lower())
    X=list(zip(*[x1,x2]))
    
    clf = joblib.load('classifier.pkl') 
    y = clf.predict(X)
    
    df["Water Source Types"] = y
    
    df.to_excel(out_file, index=False)
    

def train(training_data):

    df=pd.read_excel(training_data)
    df = df.fillna('')    
    x1=list(df["#water_tech"].str.lower())
    x2=list(df["#water_source"].str.lower())
    X=list(zip(*[x1,x2]))
    y=list(df["Categorized"])    
            
    clf = GClassifier().fit(X, y)
    joblib.dump(clf, 'classifier.pkl') 

def test(training_data):
    
    df=pd.read_excel(training_data)
    df = df.fillna('')
    x1=list(df["#water_tech"].str.lower())
    x2=list(df["#water_source"].str.lower())
    X=list(zip(*[x1,x2]))
    y=list(df["Categorized"])
    kf = KFold(len(df), n_folds=4)
    X=np.array(X)
    y=np.array(y)
    print(cross_validation.cross_val_score(GClassifier(), X, y, cv=kf))

class GClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self):
        pass
    
    def fit(self, X, y):
        trainx1, trainx2 = zip(*X)
        self.count_vect = CountVectorizer(analyzer='word', ngram_range=(1, 2))
        self.count_vect.fit(list(trainx1)+list(trainx2))
        X_train_counts1 = self.count_vect.transform(trainx1)
        X_train_counts2 = self.count_vect.transform(trainx2)
        X_train_counts = np.concatenate((X_train_counts1.toarray(),X_train_counts2.toarray()),axis=1)
        self.clf = RidgeClassifierCV().fit(X_train_counts, y)
        return self
    
    
    def predict(self, X):
        testx1, testx2 = zip(*X)
        X_test_counts1 = self.count_vect.transform(testx1)
        X_test_counts2 = self.count_vect.transform(testx2)
        X_test_counts = np.concatenate((X_test_counts1.toarray(),X_test_counts2.toarray()),axis=1)
        return self.clf.predict(X_test_counts)

if __name__ == "__main__":
    if len(sys.argv)==2:
        train(sys.argv[1])
    elif len(sys.argv)==3:
        predict(sys.argv[1], sys.argv[2])
