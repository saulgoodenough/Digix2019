#首先利用xgboost进行预测
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
def save_lgbmodel(model, file_path, best_iteration):
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
	data_train, data_test, categorical_feature = datapre.data_pre()
	train_y = data_train['label']-1
	train_x = data_train.drop(['label', 'uid'], axis=1)
	test_x = data_test[train_x.columns.tolist()]
	#train_x = np.array(train_x)
	#train_y = np.array(train_y)
	#进行训练集以及测试集的划分
	x_train, x_dev, y_train, y_dev = train_test_split(train_x, train_y, test_size=0.2)
	x_test = test_x
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
	data_train_lgb = lgb.Dataset(x_train, label=y_train, categorical_feature=categorical_feature) # 组成训练集
	data_dev_lgb = lgb.Dataset(x_dev, label=y_dev, categorical_feature=categorical_feature)
	num_round = 3000
	early_stopping_rounds = 50
	verbose_eval = 10
	lgb_model = lgb_model(config_lgb, data_train_lgb, num_round, [data_train_lgb, data_dev_lgb], early_stopping_rounds, verbose_eval)
	save_model(lgb_model, os.getcwd() + '/model')
	test_prediction_label = np.argmax(lgb_model.predict(x_test, num_iteration=lgb_model.best_iteration), axis=1)
	data_test['label'] = test_prediction_label+1
	data_final = data_test[['uid', 'label']]
	data_final.columns = ['id', 'label']
	data_final.to_csv('submission0.645.csv', index=False)








