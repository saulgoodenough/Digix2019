import numpy as np
import sklearn as sk  
from sklearn.linear_model import LogisticRegression  
from sklearn.externals import joblib
import pandas as pd

age_label_index = np.loadtxt(open("../data/age_train.csv", "rb"), delimiter=",")
age_label = age_label_index[:, 1]
total_num = len(age_label)
train_X = np.load('../data/act1hot_train.npy').item().toarray()
# logistic regression
X = train_X
y = age_label
LR = LogisticRegression(class_weight='balanced', random_state=0, solver='lbfgs', multi_class='ovr').fit(X, y)

uid_df = pd.read_csv("../data/age_test.csv", sep=',',  header=None)
uid_df.columns = ['id']
test_X = np.load('../data/act1hot_test.npy').item().toarray()

predict_label = LR.predict(test_X)
test_df = pd.DataFrame(predict_label)
test_df.columns = ['label']
print(predict_label.shape)

data_final = pd.DataFrame()
data_final['id'] = uid_df['id']
data_final['label']  = test_df['label'].astype(int)
data_final.to_csv('submission.csv', index=False)
# save the trained model
joblib.dump(LR, 'LogisticRegressione.pkl')

'''
Accuracy =  0.573173

'''