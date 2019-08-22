# coding=utf-8
# /usr/bin/env python3
'''
Author:Fuxin Jiang
Email:jiangfuxin17@mails.ucas.ac.cn
'''
#该代码主要针对usage处理之后的APP的duration还有times进行处理，选择激活数量比较多的APP的usage进行分析
import pandas as pd
from scipy import sparse
import numpy as np
import multiprocessing
def get_sparse_matrix(col_index_list, sparse_matrix):
    '''

    :param col_index:  所需要进行抽取的行数
    :param sparse_matrix: 所要进行抽取矩阵
    :return:
    '''
    sparse_matrix_generate = sparse_matrix[:, col_index_list]
    return sparse_matrix_generate
def get_sparse_matrix_sum(col_index_list, sparse_matrix):
    sparse_matrix_generate = get_sparse_matrix(col_index_list, sparse_matrix)
    #提取出加和的
    return np.ravel(np.sum(sparse_matrix_generate, axis=1))
def main1(app_class_dict, effective_app_id, sparse_matrix_train, sparse_matrix_test, useful_attri, optation):
    user_app_class_index={}
    def subpro(uni_class):
        app_class_list = app_class_dict[uni_class]
        duration_list = []
        for i in app_class_list:
            if i in effective_app_id:
                index = effective_app_id.index(i)
                duration_list.append(int(index))
        app_class_index[uni_class] = duration_list

    with multiprocessing.Manager() as manager:
        app_class_index = manager.dict()
    #进行多进程对上面的进行计算解决
        multipro = []
        #选取有用的APP类别，去除没有用的APP类别，加快速度
        #useful_attri = list(set(app_info['app_class'])-set(['合作壁纸*', '休闲娱乐', '模拟游戏', '角色游戏', '主题铃声', '策略游戏', '医疗健康', '体育射击', '电子书籍', '动作冒险']))
        for i, class_name in enumerate(useful_attri):
            #定义进程的名字
            thread_name = "thead_%d" % i
            multipro.append(multiprocessing.Process(target=subpro, name=thread_name, args=(class_name, )))
        for process in multipro:
            process.start()
        for process in multipro:
            process.join()
        print("多进程计算完毕!")

        for class_name in useful_attri:
            user_app_class_index[class_name] = app_class_index[class_name]
    sparse_matrix_train_class = np.zeros((sparse_matrix_train.shape[0], len(useful_attri)))
    sparse_matrix_test_class = np.zeros((sparse_matrix_test.shape[0], len(useful_attri)))
    for i, class_name in enumerate(useful_attri):
        sparse_matrix_train_class[:, i] = get_sparse_matrix_sum(user_app_class_index[class_name], sparse_matrix_train)
        sparse_matrix_test_class[:, i] = get_sparse_matrix_sum(user_app_class_index[class_name], sparse_matrix_test)

    np.save("train_sparse_matrix_%s_sum_app_class.npy" % optation, sparse_matrix_train_class)
    np.save("test_sparse_matrix_%s_sum_app_class.npy" % optation, sparse_matrix_test_class)
    print("结束！")


if __name__ == "__main__":

    app_info = pd.read_csv("../app_info.csv", header=None)
    app_info.columns = ['app_id', 'app_class']
    print(set(app_info['app_class']))
    app_class_to_id_dict = {}
    for class_name in set(app_info['app_class']):
        app_class_to_id_dict[class_name] = list(app_info.loc[app_info['app_class'] == class_name, 'app_id'])
    print("字典建立完毕！")
    useful_attri = list(set(app_info['app_class']) - set(['合作壁纸*', '休闲娱乐', '模拟游戏', '角色游戏', '主题铃声', '策略游戏', '医疗健康', '体育射击', '电子书籍']))
    effective_actived_app_sorted = pd.read_csv("effective_apps_duration.csv")
    effective_actived_app_sorted.columns = ['appid']
    effective_app_id = list(effective_actived_app_sorted['appid'])
    train_sparse_matrix_doration_npz = sparse.load_npz("train_sparse_matrix_duration_sum.npz")
    test_sparse_matrix_doration_npz = sparse.load_npz("test_sparse_matrix_duration_sum.npz")
    main1(app_class_to_id_dict, effective_app_id, train_sparse_matrix_doration_npz, test_sparse_matrix_doration_npz, useful_attri, 'duration')
    del train_sparse_matrix_doration_npz, test_sparse_matrix_doration_npz
    effective_actived_app_sorted = pd.read_csv("effective_apps_times.csv")
    effective_actived_app_sorted.columns = ['appid']
    effective_app_id = list(effective_actived_app_sorted['appid'])
    train_sparse_matrix_times_npz = sparse.load_npz("train_sparse_matrix_times_sum.npz")
    test_sparse_matrix_times_npz = sparse.load_npz("test_sparse_matrix_times_sum.npz")
    main1(app_class_to_id_dict, effective_app_id, train_sparse_matrix_times_npz, test_sparse_matrix_times_npz, useful_attri, 'times')
    '''
    #首先读取有效APP根据actived次数进行排序的顺序，提取3000个APP的usage信息进行分析
    effective_actived_app_sorted = pd.read_csv("../effective_apps_actived_number_sorted.csv").head(3000)
    effctive_apps_actived_list = list(effective_actived_app_sorted['appid'])
    print(effctive_apps_actived_list)
    print(len(effctive_apps_actived_list))
    #然后读取duration和times的APP信息
    usage_duration_app = pd.read_csv("effective_apps_duration.csv", header=None, names=['appid'])
    usage_times_app = pd.read_csv("effective_apps_times.csv", header=None, names=['appid'])
    effctive_apps_actived_duration_list = list(usage_duration_app['appid'])
    effctive_apps_actived_times_list = list(usage_times_app['appid'])

    duration_list = []
    times_list = []
    for i in effctive_apps_actived_list:
        index_duration = effctive_apps_actived_duration_list.index(i)
        index_times = effctive_apps_actived_times_list.index(i)
        duration_list.append(int(index_duration))
        times_list.append(int(index_times))

    train_sparse_matrix_duration_npz = sparse.load_npz("train_sparse_matrix_duration_sum.npz")
    train_sparse_matrix_duration_npz = get_sparse_matrix(duration_list, train_sparse_matrix_duration_npz)
    sparse.save_npz("train_sparse_matrix_duration_sum_3000.npz", train_sparse_matrix_duration_npz)
    print(np.shape(train_sparse_matrix_duration_npz))
    del train_sparse_matrix_duration_npz

    test_sparse_matrix_duration_npz = sparse.load_npz("test_sparse_matrix_duration_sum.npz")
    test_sparse_matrix_duration_npz = get_sparse_matrix(duration_list, test_sparse_matrix_duration_npz)
    sparse.save_npz("test_sparse_matrix_duration_sum_3000.npz", test_sparse_matrix_duration_npz)
    print(np.shape(test_sparse_matrix_duration_npz))
    del test_sparse_matrix_duration_npz

    train_sparse_matrix_times_npz = sparse.load_npz("train_sparse_matrix_times_sum.npz")
    train_sparse_matrix_times_npz = get_sparse_matrix(times_list, train_sparse_matrix_times_npz)
    sparse.save_npz("train_sparse_matrix_times_sum_3000.npz", train_sparse_matrix_times_npz)
    del train_sparse_matrix_times_npz

    test_sparse_matrix_times_npz = sparse.load_npz("test_sparse_matrix_times_sum.npz")
    test_sparse_matrix_times_npz = get_sparse_matrix(times_list, test_sparse_matrix_times_npz)
    sparse.save_npz("test_sparse_matrix_times_sum_3000.npz", test_sparse_matrix_times_npz)
    del test_sparse_matrix_times_npz
    '''





