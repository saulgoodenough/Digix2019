#import os
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
#os.environ["CUDA_VISIBLE_DEVICES"] = ""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import keras.backend as K
from keras import regularizers
from keras.layers import Input, Dense, concatenate, BatchNormalization, Dropout,Embedding, Reshape, LSTM
from keras.models import Model,load_model
from keras.utils import to_categorical
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from keras import optimizers
import runfcn_new
from scipy import sparse


DATADIR_pro = '../data/'
#WORKDIR = r'D:\Users\xyliu\digix2019'
#os.chdir(WORKDIR)
X_app_6 = sparse.load_npz(DATADIR_pro + '/test_sparse_matrix.npz')
X_bhv = np.load(DATADIR_pro + '/behav_test.npy')
X_bhv_log = np.load(DATADIR_pro + '/log_behav_test.npy')

X_bsc1 = np.load(DATADIR_pro + '/basic1_test.npy')
X_bsc2 = np.load(DATADIR_pro + '/basic2_test.npy').item()
X_act = np.load(DATADIR_pro + '/act1hot_test.npy').item()

#X_app_1 = np.loadtxt(open(DATADIR_pro+"/data_train_appclass.csv", "rb"), delimiter=",", skiprows = 1)
X_app_1 = pd.read_csv(DATADIR_pro + "/data_test_appclass.csv", sep =",")
X_app_2 = X_app_1.iloc[:, -40:].values
X_app = np.array(X_app_2).astype(int)

X_usage_time = sparse.load_npz(DATADIR_pro + 'data_norm/usage_tm_act_norm_test.npz')

X_usage_duration = sparse.load_npz(DATADIR_pro + 'data_norm/usage_dur_act_norm_test.npz')

print('X_bhv_log', type(X_bhv_log), X_bhv_log.shape)
print('X_bsc1', type(X_bsc1), X_bsc1.shape)
print('X_bsc2', type(X_bsc2), X_bsc2.shape)
print('X_act', type(X_act), X_act.shape)
print('X_app', type(X_app), X_app.shape)
print('X_app_6', type(X_app_6), X_app_6.shape)


fcn_trained_model = 'model/dnn'
model = load_model(fcn_trained_model)
'''
model.compile(loss={'out_regress': 'mean_squared_error', 'out_clf': 'categorical_crossentropy'},
                  loss_weights=[1, 1000000.],
                  metrics={'out_regress': 'mae', 'out_clf': 'acc'})
'''

results = model.predict([X_bhv_log, X_bsc1, X_bsc2, X_act, X_app, X_app_6, X_usage_time, X_usage_duration])

print(len(results))
proba_array = results[1]

class_results = np.argmax(proba_array, axis=1) + 1
print(class_results)
print(len(class_results))

x_temp = 1/np.sum(proba_array,axis=1)
x_temp = x_temp.reshape((502500, 1))

proba_array_temp = np.multiply(proba_array, x_temp)

print(proba_array_temp)
print(np.shape(proba_array_temp))

np.save('FCN_prob_testset.npy', proba_array_temp)


uid_df = pd.read_csv("../data/age_test.csv", sep=',', header=None)
uid_df.columns = ['id']
test_df = pd.DataFrame(class_results)
test_df.columns = ['label']



data_final = pd.DataFrame()
data_final['id'] = uid_df['id']
data_final['label']  = test_df['label'].astype(int)
data_final.to_csv('submission0.645.csv', index=False)






