# coding=utf-8
# /usr/bin/env python3
'''
Author:Fuxin Jiang
Email:jiangfuxin17@mails.ucas.ac.cn
'''
#对user_app_usage.py产生的文件进行处理，产生的文件包括训练集以及测试集
import pandas as pd
import json
from multiprocessing import Process
#该程序主要用于user_app_usage这个文件中训练集合和测试集合中缺失的进行处理
def missing_user_plan(dict_path, user_list, file_outpath_train):
    '''

    :param dict_path:  存储每一个用户字典格式的数据
    :param datafram:  存储训练集以及测试集的用户的一个dataframe
    :return:
    '''
    with open(dict_path, 'r') as fp:
        line = fp.readline()
        index = 0
        while line:
            line_dict = json.loads(line)
            if int(user_list[index]) != int(line_dict["uid"]):
                while (int(user_list[index]) != int(line_dict["uid"])):
                    line_dict_missing = {"a00109386_doration_max":0.0, "a00109386_doration_min":0.0, "a00109386_doration_mean":0.0, "a00109386_times_max":0.0, "a00109386_times_min":0.0, "a00109386_times_mean":0.0}
                    write_file(json.dumps(line_dict_missing), file_outpath_train)
                    index = index + 1
            index = index + 1
            del line_dict["uid"]
            write_file(json.dumps(line_dict), file_outpath_train)
            line = fp.readline()
    if index <= len(user_list):
        for i in user_list[index:]:
            line_dict_missing = {"a00109386_doration_max": 0.0, "a00109386_doration_min": 0.0, "a00109386_doration_mean": 0.0, "a00109386_times_max": 0.0, "a00109386_times_min": 0.0, "a00109386_times_mean": 0.0}
            write_file(json.dumps(line_dict_missing), file_outpath_train)
def write_file(lines, file_path):
    with open(file_path, 'a') as fw:
        lines = lines + '\n'
        fw.write(lines)


if __name__ == "__main__":
    data_train = pd.read_csv("age_train.csv", header=None)
    data_train.columns = ['uid', 'label']
    data_test = pd.read_csv("age_test.csv", header=None)
    data_test.columns = ['uid']
    data_train_user_list = list(data_train['uid'])
    data_test_user_list = list(data_test['uid'])
    del data_train, data_test
    missing_user_plan('/home/jiangfuxin/pratice_train', data_train_user_list, '/home/jiangfuxin/practice_train_out')


