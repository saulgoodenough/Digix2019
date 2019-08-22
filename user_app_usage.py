# coding=utf-8
# /usr/bin/env python3
'''
Author:Fuxin Jiang
Email:jiangfuxin17@mails.ucas.ac.cn
'''
#主要利用多线程进行加速处理数据
import pandas as pd
#首先将原来的usage里面没有用的app过滤掉
import time
import multiprocessing
import os

import numpy as np
import json
from joblib import Parallel, delayed
import multiprocessing
from scipy import sparse
from sklearn.feature_extraction import DictVectorizer
#训练集、测试集的路径
age_train_file = "../age_train.csv"
age_test_file = "../age_test.csv"
#该函数主要用于过滤没有用的app的
def filter_app_usage(file_path, effctive_apps_files):

    time_start_ = time.time()
    #user_app_usage = pd.read_csv(file_path, iterator=True, header=None, names=['id', 'app', 'duration', 'times', 'date'], usecols=['id', 'app', 'duration', 'times'])
    user_app_usage = pd.read_csv(file_path, header=None, names=['id', 'app', 'duration', 'times', 'date'], usecols=['id', 'app', 'duration', 'times'])
    effective_apps = pd.read_csv(effctive_apps_files, header=None)
    effective_apps.columns = ['app']
    effective_apps_list = list(effective_apps['app'])
    loop = True
    #最好可以用最大进程数
    '''
    thread_num = 48
    data_size = 651007719
    chunkSize = int(data_size/thread_num)+1
    
    def unipre(i, temp):
        sub_usage_app_usage_filter = temp.merge(effective_apps, on='app')
        user_app_usage_dict[i] = sub_usage_app_usage_filter
    with multiprocessing.Manager() as manager:
        user_app_usage_dict = manager.dict()
        multipro = []
        for i in range(48):
            print(i)
            temp = user_app_usage.get_chunk(chunkSize)
            # 根据appId进行merge
            thread_name = "usage_thead_%d" % i
            multipro.append(multiprocessing.Process(target=unipre, name=thread_name, args=(i, temp,)))
            # mode='a'在原来的基础进行继续写入
            # scale_usage.to_csv('/home/jiangfuxin/user_app_usage_filter.csv', mode='a', index=False, header=False)
        for process in multipro:
            process.start()
        for process in multipro:
            process.join()
        user_app_usage_filter = pd.DataFrame()
        for i in range(48):
            user_app_usage_filter = pd.concat([user_app_usage_filter, user_app_usage_dict[i]])
    '''
    user_app_usage_filter = user_app_usage.loc[user_app_usage.app.isin(effective_apps_list)]
    print(user_app_usage_filter.head())
    print(len(user_app_usage_filter))
    time_final_ = time.time()
    print("过滤有效的APP完毕!")
    print("花费时间为:")
    print(time_final_ - time_start_)
    #这步存储比较浪费时间，如果可以的话，可以不进行存储，直接存在内存当中，再进行后面的操作，不存储的话可能整个过程就980s左右，存储太浪费时间
    #user_app_usage_filter.to_csv("user_app_usage_filter", index=False)
    return user_app_usage_filter

#对过滤之后的APP进行计算
def duration_sum_times(data):
    print("开始计算duration和times的总和计算!")
    time_start_duration_sum_times = time.time()
    def fun_sum(id, groupdata):
        temp = groupdata.groupby('app')['duration', 'times'].sum()
        temp['id'] = id
        return temp
    def applyparallel(groupdata, func):
        print("开始进行并行")
        ret = Parallel(n_jobs=-1)(delayed(func)(name, group) for name, group in groupdata)
        return pd.concat(ret)
    data_group = data.groupby('id')
    del data
    data_new = applyparallel(data_group, fun_sum)
    #data_new  = data.groupby(['id', 'app']).parallel_apply(lambda x: np.sum(x[['duration', 'times']]))
    print("计算结束!")
    data_new.to_csv('user_app_usage_sum.csv')
    #这非常关键！，为了保留app的名称非常关键,后面用于存储
    data_new.reset_index(inplace=True)
    #进行训练集、测试集的划分
    data_train = pd.read_csv(age_train_file, header=None, names=['id', 'label'], usecols=['id'])
    # 提取测试集样本
    data_test = pd.read_csv(age_test_file , header=None, names=['id'])
    data_new = data_new[data_new.columns.tolist()]
    user_app_usage_train = pd.merge(data_train,data_new, how='left',on='id')
    user_app_usage_train['app'] = user_app_usage_train['app'].fillna('a00289791')
    #如果内存足够，这块可以用多线程进行训练集和测试集的
    #缺失的先按照零进行补充
    user_app_usage_train['duration'] = user_app_usage_train['duration'].fillna(int(0))
    user_app_usage_train['times'] = user_app_usage_train['times'].fillna(int(0))
    user_app_usage_train.to_csv('user_app_usage_train.csv', index=False, header=None)
    del user_app_usage_train

    user_app_usage_test = pd.merge(data_test,data_new, how='left',on='id')
    user_app_usage_test['app'] = user_app_usage_test['app'].fillna('a00289791')
    #缺失的先按照零进行补充
    user_app_usage_test['duration'] = user_app_usage_test['duration'].fillna(int(0))
    user_app_usage_test['times'] = user_app_usage_test['times'].fillna(int(0))
    user_app_usage_test.to_csv('user_app_usage_test.csv', index=False, header=None)
    del user_app_usage_test

    time_final_duration_sum_times = time.time()
    print("训练集的usage和测试集的usage已经划分好了！")
    print("花费时间为:", time_final_duration_sum_times-time_start_duration_sum_times)


def pandas_to_dict(train_csv_path, test_csv_path):
    #对doration、times进行分开存储
    print("开始进行字典存储!")
    time_start_andas_to_dict = time.time()
    def read_csv_file(file_path, file_outpath_train_doration, file_outpath_train_times):
        #每一行存储一个字典的格式, 该文件当中存放了30天内按天统计每个用户对具体某个app的累计打开次数和使用时长，格式为
        #"uid","appid","doration",times,use_date
        with open(file_path, 'r') as fp:
            uid_dict_doration = {}
            uid_dict_times = {}
            line = fp.readline()
            uid_current = line.strip().split(',')[0]
            appid = line.strip().split(',')[1]
            while line:
                line_new = line.strip().split(',')
                uid = line_new[0]
                #uid_dict_doration['uid'] = uid_current
                #uid_dict_times['uid'] = uid_current
                appid = line_new[1]
                doration = line_new[2]
                times = line_new[3]
                if uid == uid_current:
                    uid_dict_doration[str(appid)] = float(doration)
                    uid_dict_times[str(appid)] = float(times)
                if uid != uid_current:
                    #写出文件, 在文件当中写入一个json格式的一行
                    write_file(json.dumps(uid_dict_doration), file_outpath_train_doration)
                    write_file(json.dumps(uid_dict_times), file_outpath_train_times)
                    uid_current = uid
                    #uid_dict_doration['uid'] = uid_current
                    #uid_dict_times['uid'] = uid_current
                    uid_dict_doration = {}
                    uid_dict_times = {}
                uid_dict_doration[str(appid)] = float(doration)
                uid_dict_times[str(appid)] = float(times)
                line = fp.readline()
            #对最后一行进行处理，如果到了最后会出现只有一个用户，所以并不会进行输出
            uid_dict_doration[str(appid)] = float(doration)
            uid_dict_times[str(appid)] = float(times)
            write_file(json.dumps(uid_dict_doration), file_outpath_train_doration)
            write_file(json.dumps(uid_dict_times), file_outpath_train_times)

    def write_file(lines, file_path):
        with open(file_path, 'a') as fw:
            lines = lines + '\n'
            fw.write(lines)

    mulpre_1 = multiprocessing.Process(target=read_csv_file, name="train_usage_dict", args=(train_csv_path, 'user_app_usage_dict_duration_train', 'user_app_usage_dict_times_train',))
    mulpre_2 = multiprocessing.Process(target=read_csv_file, name="test_usage_dict", args=(test_csv_path, 'user_app_usage_dict_duration_test', 'user_app_usage_dict_times_test',))
    mulpre_1.start()
    mulpre_2.start()
    mulpre_1.join()
    mulpre_2.join()
    time_final_andas_to_dict = time.time()
    print("转换字典格式已经完成!")
    print("所用时间为:")
    print(time_final_andas_to_dict- time_start_andas_to_dict)

#将上面所转换的字典格式转换成稀疏矩阵存储起来

def dict2sparse(dict_file_train_duration, dict_file_test_duration, dict_file_train_times, dict_file_test_times):
    #利用多进程进行duration和times进行分开操作
    print("进行转换成稀疏矩阵!")
    def uni_dict2sparse(dict_file_train, dict_file_test, operation):
        train_dict_list = []
        with open(dict_file_train) as fp_train:
            line_train = fp_train.readline()
            while line_train:
                train_dict_list.append(json.loads(line_train.replace('\n', '')))
                line_train = fp_train.readline()

        train_dict = DictVectorizer(sparse=True)
        train_sparse_matrix = train_dict.fit_transform(train_dict_list)
        appNames = pd.Series(train_dict.feature_names_)
        appNames.to_csv('effective_apps_%s.csv' % operation, header=False, index=False)
        #np.save("train_sparse_matrix.npy", train_sparse_matrix)
        sparse.save_npz('train_sparse_matrix_%s_sum.npz' % operation, train_sparse_matrix)
        del train_dict_list
        fp_train.close()

        test_dict_list = []
        with open(dict_file_test) as fp_test:
            line_test = fp_test.readline()
            while line_test:
                test_dict_list.append(json.loads(line_test.replace('\n', '')))
                line_test = fp_test.readline()
        test_sparse_matrix = train_dict.transform(test_dict_list)
        #np.save("test_sparse_matrix.npy", test_sparse_matrix)
        sparse.save_npz("test_sparse_matrix_%s_sum.npz" % operation, test_sparse_matrix)
        fp_test.close()

    mulpre_1 = multiprocessing.Process(target=uni_dict2sparse, name="duration_usage_dict_2_sparsematrix", args=(dict_file_train_duration, dict_file_test_duration, 'duration',))
    mulpre_2 = multiprocessing.Process(target=uni_dict2sparse, name="times_usage_dict_2_sparsematrix", args=(dict_file_train_times, dict_file_test_times, 'times',))
    mulpre_1.start()
    mulpre_2.start()
    mulpre_1.join()
    mulpre_2.join()
    print("转换成稀疏矩阵完毕!")



if __name__ == "__main__":

    # 首先对原来的文件进行排序,-T是为了怕中间的tmp文件存储空间不足，而建立一个虚拟的存储空间
    try:
        os.system("rm user_app_usage_dict_*")
        print("成功删除文件!")
    except:
        pass
    time_start = time.time()
    #读取原始文件
    file_path2 = '../user_app_usage.csv'
    #过滤有用的APP
    effective_apps_file = "../effective_apps_max_features_5000.csv"
    user_app_usage_filter = filter_app_usage(file_path2, effective_apps_file)
    print("读取usage文件完毕!")

    #对APP的duration还有times进行加和处理
    duration_sum_times(user_app_usage_filter)
    del user_app_usage_filter

    #将处理好的存储一个字典
    train_file = 'user_app_usage_train.csv'
    test_file = 'user_app_usage_test.csv'
    pandas_to_dict(train_file, test_file)

    #将字典转换成稀疏矩阵
    dict_file_train_duration = 'user_app_usage_dict_duration_train'
    dict_file_test_duration = 'user_app_usage_dict_duration_test'
    dict_file_train_times = 'user_app_usage_dict_times_train'
    dict_file_times = 'user_app_usage_dict_times_test'
    dict2sparse(dict_file_train_duration, dict_file_test_duration, dict_file_train_times, dict_file_times)
    time_final = time.time()
    print("总共用时为：", time_final - time_start)



