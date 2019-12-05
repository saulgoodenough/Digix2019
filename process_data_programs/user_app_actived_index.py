# coding=utf-8
# /usr/bin/env python3
'''
Author:Fuxin Jiang
Email:jiangfuxin17@mails.ucas.ac.cn
'''
import collections
import numpy as np
import pandas as pd
def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()
    index = 1
    with open(vocab_file, "r") as fp:
        line = fp.readline()
        while line:
            token = line.strip()
            vocab[token] = index
            index += 1
            line = fp.readline()
    return vocab
#最大长度为888，通过计算发现最大长度为888。即每个人使用的最大长度为888
def user_app_actived_index(input_file, user_list, vocab_file):
    vocab_dict = load_vocab(vocab_file)
    x_list = []
    index = 0
    with open(input_file, 'r') as fp:
        line = fp.readline()
        while line:
            line_new = line.strip().split(',')
            x = [vocab_dict.get(word, 1) for word in (line_new[1]).split('#')]
            if int(line_new[0]) == user_list[index]:
                x_list.append(np.array(x))
                index = index +1
            if index == len(user_list):
                break
            line = fp.readline()
        X = np.zeros([len(x_list), 888], np.int32)
    for i, x in enumerate(x_list):
        #进行填充，对于长度小于最大长度的句子
        X[i] = np.lib.pad(x, [0, 888-len(x)], 'constant', constant_values=(0, 0))
    print(np.shape(X))
    np.save("test_user_app_actved_index.npy", X)
    return X
if __name__ == "__main__":
    input_file = "user_app_actived_sort"
    vocab_file = "effective_apps.csv"
    data_train = pd.read_csv("age_test.csv", header=None)
    data_train.columns = ['uid']
    user_list = list(data_train['uid'])
    user_app_actived_index(input_file, user_list, vocab_file)
