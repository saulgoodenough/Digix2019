# coding=utf-8
# /usr/bin/env python3
'''
Author:Fuxin Jiang
Email:jiangfuxin17@mails.ucas.ac.cn
'''
import user_app_usage_goon as goon
import pandas as pd
if __name__ == "__main__":
    data_test = pd.read_csv("age_test.csv", header=None)
    data_test.columns = ['uid']
    data_test_user_list = list(data_test['uid'])
    del data_test
    goon.missing_user_plan('/home/jiangfuxin/user_app_usage_test_9400', data_test_user_list, '/home/jiangfuxin/user_app_usage_test_9400_add_missing')