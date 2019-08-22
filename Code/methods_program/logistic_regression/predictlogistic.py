import numpy as np
import sklearn as sk
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib
import pandas as pd

uid_df = pd.read_csv("../data/age_test.csv", sep=',', header=None)
uid_df.columns = ['id']
test_X = np.load('../data/act1hot_test.npy').item().toarray()

LR = joblib.load('LogisticRegressione.pkl')
predict_label = LR.predict(test_X)
test_df = pd.DataFrame(predict_label)
test_df.columns = ['label']
print(predict_label.shape)

data_final = pd.DataFrame()
data_final['id'] = uid_df['id']
data_final['label']  = test_df['label'].astype(int)
data_final.to_csv('submission.csv', index=False)
#np.savetxt("submission.csv", test_array, delimiter=",")
