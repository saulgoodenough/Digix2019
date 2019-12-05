# coding=utf-8
# /usr/bin/env python3
'''
Author:Fuxin Jiang
Email:jiangfuxin17@mails.ucas.ac.cn
'''
'''
该代码主要用于处理对user_app_usage处理之后生成的字典存储成一个稀疏矩阵进行后面的训练
'''
import json
from scipy import sparse
from sklearn.feature_extraction import DictVectorizer
import numpy as np
def dict2sparse(dict_file_train, dict_file_test):
    train_dict_list = []

    with open(dict_file_train) as fp_train:
        line_train = fp_train.readline()
        while line_train:
            train_dict_list.append(json.loads(line_train.replace('\n', '')))
            line_train = fp_train.readline()

    train_dict = DictVectorizer(sparse=True)
    train_sparse_matrix = train_dict.fit_transform(train_dict_list)
    np.save("train_sparse_matrix.npy", train_sparse_matrix)
    sparse.save_npz("train_sparse_matrix.npz", train_sparse_matrix)
    del train_dict_list
    fp_train.close()

    test_dict_list = []
    with open(dict_file_test) as fp_test:
        line_test = fp_test.readline()
        while line_test:
            test_dict_list.append(json.loads(line_test.replace('\n', '')))
            line_test = fp_test.readline()
    test_sparse_matrix = train_dict.transform(test_dict_list)
    np.save("test_sparse_matrix.npy", test_sparse_matrix)
    sparse.save_npz("test_sparse_matrix.npz", test_sparse_matrix)
    fp_test.close()
if __name__ == "__main__":
    dict2sparse('/home/jiangfuxin/user_app_usage_new_train_9400_add_missing', '/home/jiangfuxin/user_app_usage_test_9400_add_missing')