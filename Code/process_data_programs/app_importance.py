# coding=utf-8
# /usr/bin/env python3
'''
Author:Fuxin Jiang
Email:jiangfuxin17@mails.ucas.ac.cn
'''
#改程序主要针对app的激活情况进行分析
import pandas as pd
if __name__ == "__main__":

    effective_apps = pd.read_csv("effective_apps.csv", header=None)
    effective_apps.columns = ['app_id']

    data_train = pd.read_csv("age_train.csv", header=None)
    data_train.columns = ['uid', 'label']

    data_test = pd.read_csv("age_test.csv", header=None)
    data_test.columns = ['uid']

    user_app_actived = pd.read_csv("user_app_actived.csv", header=None)
    user_app_actived.columns = ['uid', 'app_ids']

    data_train = pd.merge(data_train, user_app_actived, how='left', on='uid')

    for app_id_ in set(list(effective_apps['app_id'])):
        #print(app_id_)
        #print(set([app_id_]))
        data_train[app_id_] = data_train['app_ids'].apply(lambda x: len(set(x.strip().split('#'))&  set([app_id_])))

    data_train.drop(['app_ids'], axis=1, inplace=True)

    data_train.to_csv("app_list.csv", index=False, encoding="utf-8")




