import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import keras.backend as K
from keras import regularizers
from keras.layers import Input, Dense, concatenate, BatchNormalization, Dropout,Embedding, Reshape, LSTM, \
    Add, Multiply, Average,Concatenate, PReLU
from keras.models import Model
from keras.utils import to_categorical
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from keras import optimizers
from keras.callbacks import EarlyStopping, ModelCheckpoint
from scipy import sparse
from keras import initializers




DATADIR_pro = 'E:/Users/bzhang/huawei_age/data/'
#X_tfidf = sparse.load_npz(DATADIR_pro + 'tfidf/active_tf_idf_train.npz')
X_bhv = np.load(DATADIR_pro + '/behav_train.npy')
X_bhv_log = np.load(DATADIR_pro + '/log_behav_train.npy')
X_bsc1_temp = np.load(DATADIR_pro + '/basic1_train.npy')
X_app_number = np.load(DATADIR_pro + 'present_data/app_actived_number_train.npy')
X_bsc1 = np.concatenate((X_bsc1_temp, X_app_number), axis=1)
X_bsc2 = np.load(DATADIR_pro + '/basic2_train.npy').item().tocoo(copy=False).toarray()
#X_act = np.load(DATADIR_pro + '/act1hot_train.npy').item()
#X_app_1 = np.loadtxt(open(DATADIR_pro+"/data_train_appclass.csv", "rb"), delimiter=",", skiprows = 1)
X_app_1 = pd.read_csv(DATADIR_pro + "/data_train_appclass.csv", sep =",")
X_app_2 = X_app_1.iloc[:, -40:].values
X_app = np.array(X_app_2).astype(int)
#X_usage_time = sparse.load_npz(DATADIR_pro + 'data_norm/usage_tm_act_norm_train.npz')
#X_usage_duration = sparse.load_npz(DATADIR_pro + 'data_norm/usage_dur_act_norm_train.npz')
#X_constant = sparse.load_npz(DATADIR_pro + 'present_data/df_train_norm_sparse_876_10_constant.npz')
X_cv_max = sparse.load_npz(DATADIR_pro + 'present_data/train_actived_cv_max_features_5000.npz').tocoo(copy=False).toarray()

X = np.concatenate((X_bsc1, X_bhv_log, X_bsc2, X_app, X_cv_max), axis=1)
np.save('train_5000.npy', X)


X_bhv = np.load(DATADIR_pro + '/behav_test.npy')
X_bhv_log = np.load(DATADIR_pro + '/log_behav_test.npy')
X_bsc1_temp = np.load(DATADIR_pro + '/basic1_test.npy')
X_app_number = np.load(DATADIR_pro + 'present_data/app_actived_number_test.npy')
X_bsc1 = np.concatenate((X_bsc1_temp, X_app_number), axis=1)
X_bsc2 = np.load(DATADIR_pro + '/basic2_test.npy').item().tocoo(copy=False).toarray()
#X_act = np.load(DATADIR_pro + '/act1hot_train.npy').item()
#X_app_1 = np.loadtxt(open(DATADIR_pro+"/data_train_appclass.csv", "rb"), delimiter=",", skiprows = 1)
X_app_1 = pd.read_csv(DATADIR_pro + "/data_test_appclass.csv", sep =",")
X_app_2 = X_app_1.iloc[:, -40:].values
X_app = np.array(X_app_2).astype(int)
#X_usage_time = sparse.load_npz(DATADIR_pro + 'data_norm/usage_tm_act_norm_train.npz')
#X_usage_duration = sparse.load_npz(DATADIR_pro + 'data_norm/usage_dur_act_norm_train.npz')
#X_constant = sparse.load_npz(DATADIR_pro + 'present_data/df_train_norm_sparse_876_10_constant.npz')
X_cv_max = sparse.load_npz(DATADIR_pro + 'present_data/test_actived_cv_max_features_5000.npz').tocoo(copy=False).toarray()

X = np.concatenate((X_bsc1, X_bhv_log, X_bsc2, X_app, X_cv_max), axis=1)
np.save('test_5000.npy', X)

