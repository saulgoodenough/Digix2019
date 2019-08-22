#首先利用xgboost进行预测
#主要对一万维的数据进行分析
import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import json
import codecs
import os
import lightgbm as lgb
from sklearn.model_selection import  KFold
import datapre
from scipy import sparse
from sklearn.metrics import f1_score
def xgb_model(config, DMatrix, num_round, evallist, early_stopping_rounds, verbose_eval):
	xgb_model = xgb.train(config, DMatrix, num_round, evallist, early_stopping_rounds=early_stopping_rounds, verbose_eval = verbose_eval)
	return xgb_model
def save_model(model, file_path):
	if not os.path.exists(file_path):
		os.makedirs(file_path)
	model.save_model(file_path + '/xgb.model')
#下面定义可以进行交叉验证的xgbost模型
def xgboost_cv_model(parameters, cv_parameters, cv, train_x, train_y):
	'''

	:param parameters: 基本保持不变的参数
	:param cv_parameters: 用于交叉验证的参数
	:param cv: 几折交叉验证
	:return:
	'''
	model_gsearch = GridSearchCV(estimator=XGBClassifier(learning_rate=parameters['learning_rate'], n_estimators=parameters['n_estimator'], max_depth=parameters['max_depth'], min_child_weight=parameters['min_child_weight'], gamma=parameters['gamma'], subsample=parameters['subsample'], colsample_bytree=parameters['colsample_bytree'], objective=parameters['objective'], nthread=parameters['nthread'], scale_pos_weight=parameters['scale_pos_weight'], seed=parameters['seed']), param_grid=cv_parameters, scoring='roc_auc', n_jobs=4, iid=False, cv=cv)
	model_gsearch.fit(train_x, train_y)
	return model_gsearch
#下面定义一个未经过优化的lightgbm模型
def lgb_model(config, lgb_train, num_round, valid_sets, early_stopping_rounds, verbose_eval):
	'''

	:param config:lightgbm的基本参数
	:param lgb_train: 训练数据集
	:param num_round: 总共训练的次数
	:param valid_sets: 验证集合
	:param early_stopping_rounds:提前停止
	:return:
	'''
	lgb_mdel = lgb.train(config, lgb_train, num_boost_round = num_round, valid_sets = valid_sets, early_stopping_rounds=early_stopping_rounds, verbose_eval = verbose_eval)
	return lgb_mdel
def save_lgbmodel(model, file_path):
	if not os.path.exists(file_path):
		os.makedirs(file_path)
	model.save_model(file_path + '/lgb.model', num_iteration = model.best_iteration)
#定义加权的f1值
def f1_weighted_lgb(preds, dtrain):
	labels = dtrain.get_label()
	pred = np.argmax(preds.reshape(12,-1), axis=0)
	score = f1_score(y_true=labels, y_pred=pred, average='weighted')
	return 'f1_weighted', score, True
def f1_weighted_xgb(preds, dtrain):
	labels = dtrain.get_label()
	score = f1_score(y_true=labels, y_pred=preds, average='weighted')
	return 'f1_weighted', -score
if __name__ == "__main__":

	X_bhv = np.load('behav_train.npy')
	X_bhv_log = np.load('log_behav_train.npy')

	X_bsc1 = np.load( 'basic1_train.npy')
	X_bsc2 = np.load( 'basic2_train.npy').item()
	X_act = np.load( 'act1hot_train.npy').item()
	X_app_1 = pd.read_csv("data_train_appclass.csv", sep=",")
	X_app_2 = X_app_1.iloc[:, -40:].values
	X_app = np.array(X_app_2).astype(int)
	train_sparse_matrix_npz = sparse.load_npz("train_sparse_matrix.npz")
	test_sparse_matrix_npz = sparse.load_npz("test_sparse_matrix.npz")
	train_x = sparse.hstack((X_bhv, X_bsc1, X_bsc2, X_act, X_app, train_sparse_matrix_npz))
	print(np.shape(train_x))
	age_label_index = np.loadtxt(open("age_train.csv", "rb"), delimiter=",")
	train_y = age_label_index[:, 1]-1
	x_train, x_dev, y_train, y_dev = train_test_split(train_x, train_y, test_size=1.0/6.0)
	uid_df = pd.read_csv("age_test.csv", sep=',', header=None)
	uid_df.columns = ['id']
	X_bhv_test = np.load('behav_test.npy')
	X_bhv_log_test = np.load('log_behav_test.npy')

	X_bsc1_test = np.load('basic1_test.npy')
	X_bsc2_test = np.load('basic2_test.npy').item()
	X_act_test = np.load('act1hot_test.npy').item()
	X_app_1_test = pd.read_csv("data_test_appclass.csv", sep=",")
	X_app_2_test = X_app_1_test.iloc[:, -40:].values
	X_app_test = np.array(X_app_2_test).astype(int)

	x_test = sparse.hstack((X_bhv_test, X_bsc1_test, X_bsc2_test, X_act_test, X_app_test, test_sparse_matrix_npz))
	print(np.shape(x_train))
	print(np.shape(x_dev))
	print(np.shape(x_test))


	'''
	num_round = 5000
	early_stopping_rounds = 50
	verbose_eval = 10
	xgb_model = xgb_model(config_xgb, data_train_xgb, num_round, [(data_train_xgb, 'train'), (data_dev_xgb, 'eval')],early_stopping_rounds, verbose_eval)
	save_model(xgb_model, os.getcwd() + '/model')
	test_prediction_label_xgb = xgb_model.predict(data_test_x_xgb,ntree_limit=xgb_model.best_ntree_limit)
	data_test['recommend_mode'] = test_prediction_label_xgb
	data_final = data_test[['sid', 'recommend_mode']]
	data_final.to_csv('test_predict_xgb_new.csv', index=False)
	'''
	#lgbtm模型
	config_lgb = json.load(codecs.open('lgb_config.json', 'r', 'utf-8'))
	data_train_lgb = lgb.Dataset(x_train, label=y_train) # 组成训练集
	#data_dev_lgb = lgb.Dataset(x_dev, label=y_dev, categorical_feature=categorical_feature)
	data_dev_lgb = lgb.Dataset(x_dev, label=y_dev)
	num_round = 3000
	early_stopping_rounds = 50
	verbose_eval = 10
	lgb_model = lgb_model(config_lgb, data_train_lgb, num_round, [data_train_lgb, data_dev_lgb], early_stopping_rounds, verbose_eval)
	save_lgbmodel(lgb_model, os.getcwd() + '/model')
	test_prediction_label = np.argmax(lgb_model.predict(x_test, num_iteration=lgb_model.best_iteration), axis=1)
	test_df = pd.DataFrame(test_prediction_label+1)
	test_df.columns = ['label']
	data_final = pd.DataFrame()
	data_final['id'] = uid_df['id']
	data_final['label'] = test_df['label'].astype(int)
	data_final.to_csv('submission_1w_dimension.csv', index=False)
