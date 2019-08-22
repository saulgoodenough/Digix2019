# coding=utf-8
# /usr/bin/env python3
'''
Author:Fuxin Jiang
Email:jiangfuxin17@mails.ucas.ac.cn
'''
import pandas as pd
import json
#读取超大文件的csv，处理格式存储成文本形式
#需要对数据集进行处理，并对9401的app进行处理
effective_apps = pd.read_csv("effective_apps.csv", header=None)
effective_apps.columns = ['app_id']
#存储成有效的app列表
effective_apps_list = list(effective_apps['app_id'])
data_train = pd.read_csv("age_train.csv", header=None)
data_train.columns = ['uid', 'label']
data_test = pd.read_csv("age_test.csv", header=None)
data_test.columns = ['uid']
data_train_user_list = list(data_train['uid'])
data_test_user_list = list(data_test['uid'])
print("读取文件完毕！")
#del effective_apps, data_train, data_test
def read_csv_file(file_path, file_outpath_train, file_outpath_test):
    #每一行存储一个字典的格式, 该文件当中存放了30天内按天统计每个用户对具体某个app的累计打开次数和使用时长，格式为
    #"uid","appid","doration",times,use_date
    with open(file_path, 'r') as fp:
        doration_list = []
        times_list = []
        uid_dict = {}
        line = fp.readline()
        uid_current = line.strip('').split(',')[0]
        appid_current = line.strip('').split(',')[1]
        while line:
            #print(line)
            line_new = line.strip('').split(',')
            uid = line_new[0]
            uid_dict['uid'] = uid_current
            appid = line_new[1]
            doration = line_new[2]
            times = line_new[3]
            if appid != appid_current and uid == uid_current:
                if appid_current in effective_apps_list:
                    #print(str(appid_current))
                    uid_dict[str(appid_current)+'_doration_max'] = max(doration_list)
                    uid_dict[str(appid_current) + '_doration_min'] = min(doration_list)
                    uid_dict[str(appid_current) + '_doration_mean'] = sum(doration_list)/len(doration_list)
                    uid_dict[str(appid_current) + '_times_max'] = max(times_list)
                    uid_dict[str(appid_current) + '_times_min'] = min(times_list)
                    uid_dict[str(appid_current) + '_times_mean'] = sum(times_list) / len(times_list)
                    appid_current = appid
                    doration_list = []
                    times_list = []
                else:
                    appid_current = appid
                    doration_list = []
                    times_list = []
            if uid != uid_current:
                #写出文件, 在文件当中写入一个json格式的一行
                if int(uid_dict['uid']) in data_train_user_list:
                    write_file(json.dumps(uid_dict), file_outpath_train)
                if int(uid_dict['uid']) in data_test_user_list:
                    write_file(json.dumps(uid_dict), file_outpath_test)
                uid_current = uid
                uid_dict['uid'] = uid_current
                uid_dict = {}
            doration_list.append(float(doration))
            times_list.append(float(times))
            line = fp.readline()
        #对最后一行进行处理，如果到了最后会出现只有一个用户，所以并不会进行输出
        uid_dict[str(appid_current) + '_doration_max'] = max(doration_list)
        uid_dict[str(appid_current) + '_doration_min'] = min(doration_list)
        uid_dict[str(appid_current) + '_doration_mean'] = sum(doration_list) / len(doration_list)
        uid_dict[str(appid_current) + '_times_max'] = max(times_list)
        uid_dict[str(appid_current) + '_times_min'] = min(times_list)
        uid_dict[str(appid_current) + '_times_mean'] = sum(times_list) / len(times_list)
        if int(uid_dict['uid']) in data_train_user_list:
            write_file(json.dumps(uid_dict), file_outpath_train)
        if int(uid_dict['uid']) in data_test_user_list:
            write_file(json.dumps(uid_dict), file_outpath_test)
def write_file(lines, file_path):
    with open(file_path, 'a') as fw:
        lines = lines + '\n'
        fw.write(lines)






if __name__ == "__main__":
    read_csv_file('/home/jiangfuxin/user_app_usage', '/home/jiangfuxin/user_app_usage_new_train_9400',  '/home/jiangfuxin/user_app_usage_test_9400')
    #data_train = pd.read_csv("data_train.csv")
    #data_test = pd.read_csv("data_test.csv")
    #user_basic_info = pd.read_csv("user_basic_info.csv", header=None)
    #user_basic_info.columns = ['uid', 'gender', 'city', 'prodName', 'ramCapacity', 'ramLeftRation', 'romCapacity',
     #                          'romLeftRation', 'color', 'fontSize', 'ct', 'carrier', 'os']

    #user_app_usage = pd.read_csv("/home/jiangfuxin/user_app_usage.csv", header=None)
