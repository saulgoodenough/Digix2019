# coding=utf-8
# /usr/bin/env python3
'''
Author:Fuxin Jiang
Email:jiangfuxin17@mails.ucas.ac.cn
'''
'''
数据说明，比赛数据(脱敏后)抽取的时间范是某连续30天的数据。总体上看，训练分为训练集数据文件、测试集数据文件、用户基本特征数据集、用户行为类汇总特征
数据集、用户激活过的app列表、30天的APP使用日志、APP类别元数据
age_train.csv代表训练样本，各字段之间由逗号隔开 1代表小于18岁、2代表19-23周岁、3代表24-34岁、4代表35-44岁、5代表45-54岁、6代表大于等于55周岁
训练数据总共2010000，测试数据502500
'''
'''
用户基本特征数据集user_basic_info.csv每一行代表一个用户的基本信息，包含用户人口属性、设备基本属性、各字段之间由逗号分隔，格式为:
"uld, gender, city, prodName, ramCapacity, ramLeftRation, romCapacity, romLeftRation, color, fontSize, ct,carrier, os "
用户标识（uId） 匿名化处理后的用户唯一标识（ID取值从1000001开始，依次递增）
性别（gender） 男/女（取值空间0,1）
常住地（city） 如深圳市、南京市等（匿名化处理，实际取值c001，c002….）
手机型号（prodName） 如mate10、honor 10等（匿名化处理，实际取值p001、p002……）
手机ram容量（ramCapacity） 手机ram的大小，以G为单位
ram剩余容量占比（ramLeftRation） 手机剩余的容量占总容量的比例
rom容量（romCapacity） 手机rom的大小，以G为单位
rom剩余容量占比（romLeftRation） 手机剩余rom容量占总rom容量的比例
手机颜色（color） 手机机身的颜色
字体大小（fontSize） 手机设置的字体大小
上网类型（ct） 2G/3G/4G/WIFI
移动运营商（carrier） 移动/联通/电信/其他
手机系统版本（os）AndroId操作系统的版本号
总共2512500条
'''
'''
用户行为类汇总特征数据集user_behavior_info.csv每行代表一个用户的行为类信息,包含对设备的使用行为汇总数据。
用户标识（uId） 匿名化处理后的用户唯一标识（ID取值从1000001开始，依次递增）
开机次数（bootTimes） 一段时间内(30天)手机的总开机次数
手机A特性使用次数（AFuncTimes） 一段时间内(30天) 手机A特性使用次数
手机B特性使用次数（BFuncTimes） 一段时间内(30天) 手机B特性使用次数
手机C特性使用次数（CFuncTimes） 一段时间内(30天) 手机C特性使用次数
手机D特性使用次数（DFuncTimes） 一段时间内(30天) 手机D特性使用次数
手机E特性使用次数（EFuncTimes） 一段时间内(30天) 手机E特性使用次数
手机F特性使用次数（FFuncTimes） 一段时间内(30天) 手机F特性使用次数
手机G特性使用情况（FFuncSum）   一段时间内(30天)G特性使用情况（数值)
总共2512500条
'''
'''
用户的激活APP列表文件user_app_actived.csv 每一行代表一条用户激活app的记录(APP激活的含义为用户安装并使用该APP)。特征文件格式为:
"uld, appld# appld# appld# appld# appld......"uld为用户标识，appld为app应用的唯一标识，多个app以"#"分隔
用户标识（uId） 匿名化处理后的用户唯一标识（ID取值从1000001开始，依次递增）
应用标识（appId） 匿名化处理后的app唯一标识
总共2512500条
'''
'''
app使用行为日志文件user_app_usage.csv存放了30天内按天统计每个用户对具体某个app的累计打开次数和使用时长，
用户标识（uId） 匿名化处理后的用户唯一标识（ID取值从1000001开始，依次递增）
应用标识（appId） 匿名化处理后的app唯一标识
使用时长（duration） 1天内用户对某app的累计使用时长
打开次数（times） 1天内用户对某app的累计打开次数
使用日期（use_date） 用户对某app的使用日期
总共651007719条
'''

'''
app对应类别文件app_info.csv每一行代表一条app的信息，格式如下:
应用标识（appId） appId为app应用的唯一标识
应用类型（category） app所属的应用类型
总共188864条
'''
import pandas as pd
from collections import Counter
def data_pre():

    data_train = pd.read_csv("age_train.csv", header=None)
    data_train.columns = ['uid', 'label']

    data_test = pd.read_csv("age_test.csv", header=None)
    data_test.columns = ['uid']

    user_basic_info = pd.read_csv("user_basic_info.csv", header=None)
    user_basic_info.columns = ['uid', 'gender', 'city', 'prodName', 'ramCapacity', 'ramLeftRation', 'romCapacity',
                               'romLeftRation', 'color', 'fontSize', 'ct', 'carrier', 'os']
    prodName_mapping = {label: idx for idx, label in enumerate(set(user_basic_info['prodName']))}
    user_basic_info['prodName'] = user_basic_info['prodName'].map(prodName_mapping)

    city_mapping = {label: idx for idx, label in enumerate(set(user_basic_info['city']))}
    user_basic_info['city'] = user_basic_info['city'].map(city_mapping)

    carrier_mapping = {label: idx for idx, label in enumerate(set(user_basic_info['carrier']))}
    user_basic_info['carrier'] = user_basic_info['carrier'].map(carrier_mapping)

    color_mapping = {label: idx for idx, label in enumerate(set(user_basic_info['color']))}
    user_basic_info['color'] = user_basic_info['color'].map(color_mapping)

    ct_mapping = {label: idx for idx, label in enumerate(set(user_basic_info['ct']))}
    user_basic_info['ct'] = user_basic_info['ct'].map(ct_mapping)


    user_behavior_info = pd.read_csv("user_behavior_info.csv", header=None)
    user_behavior_info.columns = ['uid', 'bootTimes', 'AFuncTimes', 'BFuncTimes', 'CFuncTimes', 'DFuncTimes',
                                  'EFuncTimes', 'FFuncTimes', 'GFuncTimes']

    app_info = pd.read_csv("app_info.csv", header=None)
    app_info.columns = ['app_id', 'app_class']
    print(set(app_info['app_class']))
    app_class_to_id_dict = {}
    for class_name in set(app_info['app_class']):
        app_class_to_id_dict[class_name] = list(app_info.loc[app_info['app_class'] == class_name, 'app_id'])
    print("字典建立完毕！")

    user_app_actived = pd.read_csv("user_app_actived.csv", header=None)
    user_app_actived.columns = ['uid', 'app_ids']


    for class_name in set(app_info['app_class']):
        user_app_actived[class_name] = user_app_actived['app_ids'].apply(
            lambda x: len(set(x.strip().split('#')) & set(app_class_to_id_dict[class_name])))
        
    user_app_actived.drop(['app_ids'], axis=1, inplace=True)
    
    data_train = pd.merge(data_train, user_basic_info, how='left', on='uid')
    data_train = pd.merge(data_train, user_behavior_info, how='left', on='uid')
    data_train = pd.merge(data_train, user_app_actived, how='left', on='uid')
    data_test = pd.merge(data_test, user_basic_info, how='left', on='uid')
    data_test = pd.merge(data_test, user_behavior_info, how='left', on='uid')
    data_test = pd.merge(data_test, user_app_actived, how='left', on='uid')
    #返回分类属性
    categorical_feature = ['gender', 'city', 'prodName', 'color', 'ct', 'carrier']

    data_train.to_csv("data_train.csv", index=False, encoding="utf-8")
    data_test.to_csv("data_test.csv", index=False, encoding="utf-8")

    return data_train, data_test, categorical_feature

if __name__ == "__main__":
    data_train, data_test ,categorical_feature = data_pre()
    print(data_train.head())
    print(data_test.head())
    app_info = pd.read_csv("app_info.csv", header = None)
    app_info.columns = ['app_id', 'app_class']
    print(set(app_info['app_class']))
    app_class_to_id_dict = {}
    for class_name in set(app_info['app_class']):
        app_class_to_id_dict[class_name] = list(app_info.loc[app_info['app_class'] == class_name, 'app_id'])
    print("字典建立完毕！")

    user_app_actived = pd.read_csv("user_app_actived.csv", header = None)
    user_app_actived.columns = ['uid', 'app_ids']

    for class_name in set(app_info['app_class']):
        user_app_actived[class_name] = user_app_actived['app_ids'].apply(lambda x :len(set(x.strip().split('#'))&set(app_class_to_id_dict[class_name])))
    print(user_app_actived.head())
