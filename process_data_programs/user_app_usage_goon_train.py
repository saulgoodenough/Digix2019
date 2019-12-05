# coding=utf-8
# /usr/bin/env python3
'''
Author:Fuxin Jiang
Email:jiangfuxin17@mails.ucas.ac.cn
'''
import user_app_usage_goon as goon
import pandas as pd
if __name__ == "__main__":
    data_train = pd.read_csv("age_train.csv", header=None)
    data_train.columns = ['uid', 'label']
    data_train_user_list = list(data_train['uid'])
    del data_train
    goon.missing_user_plan('/home/jiangfuxin/user_app_usage_new_train_9400', data_train_user_list, '/home/jiangfuxin/user_app_usage_new_train_9400_add_missing')