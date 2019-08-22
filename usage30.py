# step 2
'''
思路:
    第一步：将top_n+1维appId从 step 1 生成的文件top_n.csv提取出来
    第二步：分块读取usage文件，根据appId进行merge，分块保存
    第三步：由于数据量巨大，将步骤二结果按30天分别保存
'''
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import datetime
from scipy import sparse
from scipy.sparse import csr_matrix, hstack
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
max_abs_scaler  = preprocessing.MaxAbsScaler()
import gc
#查看文件列表
import os
print(os.listdir("./input"))
datadir = './input'
out_datadir = './sparse_30days'
# 选出排名前top_n的app
top_n = 9400 # ！！！！！！！！！！！，实际上选择了top_n+1个app
print('top_n的大小是：',top_n)
df_train = pd.read_csv(os.path.join(datadir,'age_train.csv'),header=None, names=['uId','age_group'],dtype = {'uId':np.str}
                               ,index_col = 'uId')
df_test = pd.read_csv(os.path.join(datadir,'age_test.csv'),names=['uId'],dtype = {'uId':np.str},index_col = 'uId')
df_train['trainrow'] = np.arange(df_train.shape[0])
df_test['testrow'] = np.arange(df_test.shape[0])
df_train.index = df_train.index.astype(np.str)
df_test.index = df_test.index.astype(np.str)

# 第一步：将top_n+1维appId从 step 1 生成的文件top_n.csv提取出来，注意路径,用于索引
top_n_app = pd.read_csv('top_'+str(top_n)+'.csv', header=None,names=['appId'])
top_n_app_list = top_n_app.appId.tolist()
dictionary = dict(zip(top_n_app_list,list(range(len(top_n_app_list)))))
# 第二步：分块读取usage文件，根据appId进行merge，分块保存
starttime = datetime.datetime.now()
user_app_usage = pd.read_csv(os.path.join(datadir,'user_app_usage.csv'), header=None
                             , names=['uId','appId','duration','times','use_date'])

user_app_usage_top_n = user_app_usage.loc[user_app_usage.appId.isin(top_n_app_list)]# top_n + 1

endtime = datetime.datetime.now()
print('过滤app耗时{}s'.format((endtime - starttime).seconds))

# 第三步：由于数据量巨大，将步骤二结果按30天分别保存
datalist=['2019-02-28']
datalist1= [str('2019-03-0')+str(i) for i in range(1,10)]
datalist2 = [str('2019-03-')+str(i) for i in range(10,30)]
datalist.extend(datalist1)
datalist.extend(datalist2)
appIds = top_n + 1
i = 0
for data_name in datalist:
        print(data_name)
        temp_1 = user_app_usage_top_n.loc[(user_app_usage_top_n['use_date']==data_name)]
        # 将times异常值去掉
        temp_1.drop(temp_1.loc[(temp_1['times'] > temp_1.times.quantile(0.99) + 0.1)].index, inplace=True)
        temp_1.reset_index(drop=True, inplace=True)
        # 将appId映射成label，保证每一天的列对应相同的app
        temp_1['appId_category'] = temp_1['appId'].map(dictionary)
        temp_1 = temp_1.dropna().reset_index()
        temp_1 = temp_1.set_index('uId')

        temp_1.index = temp_1.index.astype(np.str)

        temp_1_train = temp_1.merge(df_train, on='uId')
        temp_1_test = temp_1.merge(df_test, on='uId')

        temp_1_train_duration = csr_matrix((temp_1_train['duration'].values, (temp_1_train.trainrow, temp_1_train.appId_category)),shape=(df_train.shape[0], appIds))
        temp_1_test_duration = csr_matrix((temp_1_test['duration'].values, (temp_1_test.testrow, temp_1_test.appId_category)),shape=(df_test.shape[0], appIds))
        temp_1_train_times = csr_matrix((temp_1_train['times'].values, (temp_1_train.trainrow, temp_1_train.appId_category)),shape=(df_train.shape[0], appIds))
        temp_1_test_times = csr_matrix((temp_1_test['times'].values, (temp_1_test.testrow, temp_1_test.appId_category)),shape=(df_test.shape[0], appIds))

        if i == 0:
            train_duration = temp_1_train_duration
            test_duration = temp_1_test_duration

            train_times = temp_1_train_times
            test_times = temp_1_test_times
        else:
            train_duration += temp_1_train_duration
            test_duration += temp_1_test_duration

            train_times += temp_1_train_times
            test_times += temp_1_test_times
        i+=1
sparse.save_npz(os.path.join(out_datadir, 'train_duration30day.npz'), train_duration)  # 保存
sparse.save_npz(os.path.join(out_datadir, 'test_duration30day.npz'), test_duration)  # 保存
sparse.save_npz(os.path.join(out_datadir, 'train_times30day.npz'), train_times)  # 保存
sparse.save_npz(os.path.join(out_datadir, 'test_times30day.npz'), test_times)  # 保存

endtime1 = datetime.datetime.now()
print('usage拆分30个文件耗时{}s'.format((endtime1 - starttime).seconds))