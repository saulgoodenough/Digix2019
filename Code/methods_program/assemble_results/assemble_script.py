import numpy as np
import pandas as pd
from collections import Counter


def assemble_age(df_list):
    df_assemble = pd.DataFrame()
    df_assemble['id'] = df_list[0]['id']
    df_assemble['label'] = df_list[0]['label']
    method_num = len(df_list)
    user_num = len(df_assemble.index)
    for j in range(user_num):
        com_list = []
        for df_method in df_list:
            label_temp = df_method.label[j]
            com_list.append(label_temp)
        label_counts = Counter(com_list)
        most_common = label_counts.most_common(1)
        most_common_label = most_common[0][0]
        most_common_freq = most_common[0][1]
        if most_common_freq >= 4:
            df_assemble.label[j] = most_common_label
    return df_assemble


if __name__ == '__main__':
    file_fcn_dense = './fcn_dense/submission.csv'
    df_fcn_dense = pd.read_csv(file_fcn_dense, sep=',')
    file_fcn_1 = './fcn_1/submission.csv'
    df_fcn_1 = pd.read_csv(file_fcn_1, sep=',')
    file_fcn_2 = './fcn_2/submission.csv'
    df_fcn_2 = pd.read_csv(file_fcn_2, sep=',')
    file_rf = './lgb_2/submission.csv'
    df_rf = pd.read_csv(file_rf, sep=',')
    file_lgb = './lgb/submission.csv'
    df_lgb = pd.read_csv(file_lgb, sep=',')
    file_lgb1 = './lgb_1/submission.csv'
    df_lgb1 = pd.read_csv(file_lgb1, sep=',')
    file_logistic = './logistic/submission.csv'
    df_logistic = pd.read_csv(file_logistic, sep=',')
    df_list = [df_fcn_dense, df_rf, df_fcn_1,df_fcn_2,  df_lgb1,  df_lgb, df_logistic]
    df_assemble = assemble_age(df_list)
    df_assemble.to_csv('./fcn_lgb_log/submission.csv', index=False)


