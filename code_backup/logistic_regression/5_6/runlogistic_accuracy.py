import numpy as np
import sklearn as sk  
from sklearn.linear_model import LogisticRegression  
from sklearn.externals import joblib


age_label_index = np.loadtxt(open("../data/age_train.csv", "rb"), delimiter=",")
age_label = age_label_index[:, 1]
total_num = len(age_label)
train_num = int(5/6 * total_num)
train_X = np.load('../data/act1hot_train.npy').item().toarray()

# logistic regression

vali_X = train_X[train_num:,:]
vali_y = age_label[train_num:]
LR =  joblib.load('LogisticRegressione.pkl')
LR.predict(vali_X)  
print('Accuracy = ', round(LR.score(vali_X,vali_y), 6))

# save the trained model
joblib.dump(LR, 'LogisticRegressione.pkl')

'''
Accuracy =  0.573173

'''