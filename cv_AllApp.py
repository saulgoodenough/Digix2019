# coding=utf-8
# /usr/bin/env python3
'''
Author:Fuxin Jiang
Email:jiangfuxin17@mails.ucas.ac.cn
'''
#主要对app进行cv处理，针对user_app_actived
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from scipy import sparse
import time
#需要四分钟左右,就选择这5000个APP吧
def app_cv(data_train, data_test):
    '''
    :param max_features:
    :return:
    '''
    print("开始进行处理！")
    data_train_new = data_train.copy()
    data_test_new = data_test.copy()
    data_train_new["app_token"] = data_train_new['app_ids'].apply(lambda x : x.replace("#", ' '))
    data_test_new["app_token"] = data_test_new['app_ids'].apply(lambda x: x.replace("#", ' '))
    #选出出现次数排名前5000的APP，可以自己进行调换
    cv = CountVectorizer()
    cv.fit(data_train_new["app_token"])
    train_app_actived_cv = cv.transform(data_train_new["app_token"])
    test_app_actived_cv = cv.transform(data_test_new["app_token"])
    print(np.shape(train_app_actived_cv))
    print(np.shape(test_app_actived_cv))
    appNames = pd.Series(cv.get_feature_names())
    #存储APP的名字(有效APP的名字)
    appNames.to_csv('effective_apps_allapp.csv', header=False, index=False)
    sparse.save_npz("train_actived_cv_allapp.npz", train_app_actived_cv)
    sparse.save_npz("test_actived_cv_allapp.npz", test_app_actived_cv)

#计算每一个APP计算的激活的总个数，然后按照激活的总个数进行排序，用于后面的使用时间进行处理用
def app_actived_totalnum(active_sparse_matrix, effective_apps_file):

    app_actived_matrix = sparse.load_npz(active_sparse_matrix).toarray()
    one_app_actived_number = app_actived_matrix.sum(axis=0)
    effective_app = pd.read_csv(effective_apps_file, sep=',', header=None)
    effective_app.columns = ['appid']
    effective_app['app_actived_number'] = one_app_actived_number
    effective_app.sort_values("app_actived_number", ascending=False, inplace=True)
    effective_app.to_csv("effective_apps_allapp_sorted.csv", index=False)

if __name__ == "__main__":

    time_start = time.time()
    data_train = pd.read_csv("age_train.csv", header=None)
    data_train.columns = ['uid', 'label']
    data_test = pd.read_csv("age_test.csv", header=None)
    data_test.columns = ['uid']
    user_app_actived = pd.read_csv("user_app_actived.csv", header=None)
    user_app_actived.columns = ['uid', 'app_ids']
    data_train = pd.merge(data_train, user_app_actived, how='left', on='uid')
    data_test = pd.merge(data_test, user_app_actived, how='left', on='uid')
    app_cv(5000, data_train, data_test)
    time_final = time.time()
    print("cv过程所用时间为:", time_final-time_start)

    
    print("提取APP的总激活次数!")
    '''
    effective_apps_file = 'effective_apps_max_features_5000.csv'
    active_sparse_matrix = "train_actived_cv_max_features_5000.npz"
    app_actived_totalnum(active_sparse_matrix, effective_apps_file)
    '''