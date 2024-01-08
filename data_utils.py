# -*- coding: utf-8 -*-
"""
对GUI.py进行操作补充
@author: wangxuechao
"""
import pandas as pd
import matplotlib.pyplot as plt
import os
import warnings
import argparse
import numpy
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2, f_classif, mutual_info_classif
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier

import numpy as np
import argparse
import warnings
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from lightgbm.sklearn import LGBMClassifier
from sklearn import metrics
import joblib


plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']


def lack_processing(data):
    '''
    对不含有复发时间的数据处理
    缺失值填充：通过对最初原始数据的分析，发现NAN均在放疗或化疗两列，可以用2进行替换；
    针对性处理：删除随访时间和生存状态，删除对“放疗”或“化疗”未知数据，删除因其他死因死亡数据
    '''
    data.dropna(axis=0, how='all', inplace=True) # 删除全为NAN的行
    data.dropna(axis=1, how='all', inplace=True) # 删除全为NAN的列
    data.fillna(value=2,inplace=True) #将剩余NAN换为2
    data.drop(['随访时间','生存状态'], axis=1, inplace=True)
    # 针对性处理：删除对“放疗”或“化疗”未知数据，删除因其他死因死亡数据
    data = data[~data['放疗或粒子'].isin([2])]
    data = data[~data['化疗'].isin([2])]
    data = data[~data['全因死亡'].isin([2])]
    data.loc[data.M分期 == 1, '远处转移'] = 2
    data.drop('M分期', axis=1, inplace=True)
    return data


def lack_processing_predictive(data):
    '''
    对含有复发时间的数据处理
    缺失值填充：通过对最初原始数据的分析，发现NAN均在放疗或化疗两列，可以用2进行替换；
    针对性处理：删除对“放疗”或“化疗”未知数据，删除存活、因其他死因死亡的数据。
    '''
    data.dropna(axis=0, how='all', inplace=True) # 删除全为NAN的行
    data.dropna(axis=1, how='all', inplace=True) # 删除全为NAN的列
    data.fillna(value=2,inplace=True) #将剩余NAN换为2

    # 针对性处理：删除对“放疗”或“化疗”未知数据，删除因其他死因死亡数据
    data = data[~data['放疗或粒子'].isin([2])]
    data = data[~data['化疗'].isin([2])]

    data = data[data['全因死亡'].isin([1])] #只保留因病死亡的数据

    data = data.drop(['局部复发','颈部复发','远处转移','放疗或粒子','化疗','全因死亡'], axis=1)

    return data




'''############################################################################'''
# 单特征分布统计
def SEX(path, data):
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    # 性别
    sex = {'男': 0, '女': 0}
    for s in data['性别']:
        if s == 1:
            sex['男'] += 1
        if s == 2:
            sex['女'] += 1
    labels = sex.keys()
    fraces = sex.values()
    explode = [0.05, 0]
    plt.pie(x=fraces, labels=labels, autopct='%0.2f%%', explode=explode, )
    plt.title('性别比例')
    # plt.show()
    plt.savefig(os.path.join(path, 'sex.png'), dpi=100, bbox_inches = 'tight')
    plt.close()

def AGE(path,data):
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    # 年龄
    age_dict = data['年龄'].value_counts().to_dict()
    x = age_dict.keys()
    y = age_dict.values()
    plt.bar(x, y)
    plt.title('年龄分布')
    plt.xlabel('年龄')
    plt.ylabel('人数')
    # plt.show()
    plt.savefig(os.path.join(path, 'age.png'), dpi=100, bbox_inches = 'tight')
    plt.close()

def RADIOTHERAPY(path,data):
    # 是否接受过放疗
    N = {'N-已放疗': 0, 'N-无放疗': 0, 'N-未知': 0}
    for i in data['放疗或粒子']:
        if i == 0:
            N['N-无放疗'] += 1
        elif i == 1:
            N['N-已放疗'] += 1
        else:
            N['N-未知'] += 1
    N_labels = N.keys()
    N_fraces = N.values()
    explode = [0.1, 0, 0]
    plt.pie(x=N_fraces, labels=N_labels, autopct='%0.2f%%', explode=explode,
            textprops={'fontsize': 8, 'color': 'black'})
    plt.title('放疗（或粒子）分布')
    # plt.show()
    plt.savefig(os.path.join(path, 'radiotherapy.png'), dpi=100, bbox_inches = 'tight')
    plt.close()

def CHEMOTHERAPY(path,data):
    # 是否接受过化疗chemotherapy
    N = {'N-已化疗': 0, 'N-无化疗': 0, 'N-未知': 0}
    for i in data['化疗']:
        if i == 0:
            N['N-无化疗'] += 1
        elif i == 1:
            N['N-已化疗'] += 1
        else:
            N['N-未知'] += 1
    N_labels = N.keys()
    N_fraces = N.values()
    explode = [0.1, 0, 0]
    plt.pie(x=N_fraces, labels=N_labels, autopct='%0.2f%%', explode=explode,
            textprops={'fontsize': 8, 'color': 'black'})
    plt.title('化疗分布')
    # plt.show()
    plt.savefig(os.path.join(path, 'chemotherapy.png'), dpi=100, bbox_inches = 'tight')
    plt.close()

def T_TNM(path,data):
    #TNM分期-T
    T = {'T-1': 0, 'T-2': 0, 'T-3': 0, 'T-4': 0}
    for i in data['T分期']:
        if i == 1:
            T['T-1'] += 1
        if i == 2:
            T['T-2'] += 1
        if i == 3:
            T['T-3'] += 1
        if i == 4:
            T['T-4'] += 1
    T_labels = T.keys()
    T_fraces = T.values()
    explode = [0, 0.05, 0, 0]
    plt.pie(x=T_fraces, labels=T_labels, autopct='%0.1f%%', explode=explode,
            textprops={'fontsize': 8, 'color': 'black'})
    plt.title('TNM标准分类-T分期')
    # plt.show()
    plt.savefig(os.path.join(path, 'T.png'), dpi=100, bbox_inches = 'tight')
    plt.close()

def N_TNM(path,data):
    N = {'N-0': 0, 'N-1': 0, 'N-2': 0, 'N-3': 0}
    for i in data['N分期']:
        if i == 0:
            N['N-0'] += 1
        if i == 1:
            N['N-1'] += 1
        if i == 2:
            N['N-2'] += 1
        if i == 3:
            N['N-3'] += 1
    N_labels = N.keys()
    N_fraces = N.values()
    explode = [0.1, 0, 0, 0]
    plt.pie(x=N_fraces, labels=N_labels, autopct='%0.1f%%', explode=explode,
            textprops={'fontsize': 8, 'color': 'black'})
    plt.title('TNM标准分类-N分期')
    # plt.show()
    plt.savefig(os.path.join(path, 'N.png'), dpi=100, bbox_inches = 'tight')
    plt.close()

def M_TNM(path,data):
    M = {'M-无转移': 0, 'M-已转移': 0}
    for i in data['M分期']:
        if i == 0:
            M['M-无转移'] += 1
        if i == 1:
            M['M-已转移'] += 1
    M_labels = M.keys()
    M_fraces = M.values()
    explode = [0.1, 0]
    plt.pie(x=M_fraces, labels=M_labels, autopct='%0.1f%%', explode=explode,
            textprops={'fontsize': 8, 'color': 'black'})
    plt.title('TNM标准分类-M分期')
    # plt.show()
    plt.savefig(os.path.join(path, 'M.png'), dpi=100, bbox_inches = 'tight')
    plt.close()

def TIME(path,data):
    #随访时间
    time_dict = data['随访时间'].value_counts().to_dict()
    x = time_dict.keys()
    y = time_dict.values()
    plt.bar(x, y)
    plt.title('随访时间分布')
    plt.xlabel('时间/月')
    plt.ylabel('人数')
    # plt.show()
    plt.savefig(os.path.join(path, 'time.png'), dpi=100, bbox_inches = 'tight')
    plt.close()

def LOCAL_RELAPSE(path,data):
    #局部有无复发
    L = {'L-无复发': 0, 'L-已复发': 0}
    for i in data['局部复发']:
        if i == 0:
            L['L-无复发'] += 1
        if i == 1:
            L['L-已复发'] += 1
    L_labels = L.keys()
    L_fraces = L.values()
    explode = [0.1, 0]
    plt.pie(x=L_fraces, labels=L_labels, autopct='%0.1f%%', explode=explode,
            textprops={'fontsize': 8, 'color': 'black'})
    plt.title('局部是否复发')
    # plt.show()
    plt.savefig(os.path.join(path, 'local.png'), dpi=100, bbox_inches = 'tight')
    plt.close()

def NECK_RELAPSE(path,data):
    #颈部有无复发
    N = {'N-无复发': 0, 'N-已复发': 0}
    for i in data['颈部复发']:
        if i == 0:
            N['N-无复发'] += 1
        if i == 1:
            N['N-已复发'] += 1
    N_labels = N.keys()
    N_fraces = N.values()
    explode = [0.1, 0]
    plt.pie(x=N_fraces, labels=N_labels, autopct='%0.1f%%', explode=explode,
            textprops={'fontsize': 8, 'color': 'black'})
    plt.title('颈部是否复发')
    # plt.show()
    plt.savefig(os.path.join(path, 'neck.png'), dpi=100, bbox_inches = 'tight')
    plt.close()

def TRANSFORM(path,data):
    #是否远处转移
    N = {'术前无转移、术后无转移': 0, '术前无转移、术后有转移': 0, '术前有转移': 0}
    for i, d in enumerate(data['远处转移']):
        if d == 0:
            N['术前无转移、术后无转移'] += 1
        if d == 1:
            if data['M分期'][i] == 0:
                N['术前无转移、术后有转移'] += 1
            if data['M分期'][i] == 1:
                N['术前有转移'] += 1
    N_labels = N.keys()
    N_fraces = N.values()
    explode = [0.1, 0, 0]
    plt.pie(x=N_fraces, labels=N_labels, autopct='%0.1f%%', explode=explode,
            textprops={'fontsize': 8, 'color': 'black'})
    plt.title('远处转移')
    # plt.show()
    plt.savefig(os.path.join(path, 'transform.png'), dpi=100, bbox_inches = 'tight')
    plt.close()

def AREA(path,data):
    # 发病部位  1-腮腺；2-颌下腺；3-舌下腺+口底；4-颚；5-磨牙后区；6-颊；7-舌；8-唇；9-上颌；10-其他（下颌骨等）
    area = {'腮腺': 0, '颌下腺': 0, '舌下腺+口底': 0, '颚': 0, '磨牙后区': 0, '颊': 0, '舌': 0, '唇': 0, '上颌': 0, '其他': 0}
    for i in data['发病部位']:
        if i == 1:
            area['腮腺'] += 1
        if i == 2:
            area['颌下腺'] += 1
        if i == 3:
            area['舌下腺+口底'] += 1
        if i == 4:
            area['颚'] += 1
        if i == 5:
            area['磨牙后区'] += 1
        if i == 6:
            area['颊'] += 1
        if i == 7:
            area['舌'] += 1
        if i == 8:
            area['唇'] += 1
        if i == 9:
            area['上颌'] += 1
        if i == 10:
            area['其他'] += 1
    labels = area.keys()
    fraces = area.values()
    explode = [0.05, 0, 0, 0.05, 0, 0, 0, 0, 0, 0]
    plt.pie(x=fraces, labels=labels, autopct='%0.1f%%', explode=explode,
            textprops={'fontsize': 10, 'color': 'black'})
    plt.title('发病部位分布')
    # plt.show()
    plt.savefig(os.path.join(path, 'area.png'), dpi=100, bbox_inches = 'tight')
    plt.close()

def CLASSES(path,data):
    # 病理类型  1a-高分化粘表；1b-中分化粘表；1c-低分化粘表；2-腺样囊性癌；3-癌在多形性腺瘤中；4-非特异性腺癌；5-腺泡细胞癌；6-肌上皮癌；7-多型性腺癌；8-基底细胞腺癌；9-唾液腺导管癌；10-鳞状细胞癌；11-淋巴上皮癌；12-上皮-肌上皮癌；13-嗜酸细胞腺癌；14-透明细胞癌；15-其他
    area = {'高分化粘表': 0, '中分化粘表': 0, '低分化粘表': 0, '腺样囊性癌': 0, '癌在多形性腺瘤中': 0,
            '非特异性腺癌': 0, '腺泡细胞癌': 0, '肌上皮癌': 0, '多型性腺癌': 0, '基底细胞腺癌': 0,
            '唾液腺导管癌': 0, '鳞状细胞癌': 0, '淋巴上皮癌': 0, '上皮-肌上皮癌': 0, '嗜酸细胞腺癌': 0,
            '透明细胞癌': 0, '其他': 0}
    for i in data['病理类型']:
        if i == '1a':
            area['高分化粘表'] += 1
        if i == '1b':
            area['中分化粘表'] += 1
        if i == '1c':
            area['低分化粘表'] += 1
        if i == 2:
            area['腺样囊性癌'] += 1
        if i == 3:
            area['癌在多形性腺瘤中'] += 1
        if i == 4:
            area['非特异性腺癌'] += 1
        if i == 5:
            area['腺泡细胞癌'] += 1
        if i == 6:
            area['肌上皮癌'] += 1
        if i == 7:
            area['多型性腺癌'] += 1
        if i == 8:
            area['基底细胞腺癌'] += 1
        if i == 9:
            area['唾液腺导管癌'] += 1
        if i == 10:
            area['鳞状细胞癌'] += 1
        if i == 11:
            area['淋巴上皮癌'] += 1
        if i == 12:
            area['上皮-肌上皮癌'] += 1
        if i == 13:
            area['嗜酸细胞腺癌'] += 1
        if i == 14:
            area['透明细胞癌'] += 1
        if i == 15:
            area['其他'] += 1
    labels = area.keys()
    fraces = area.values()
    explode = [0.05, 0, 0, 0.05, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    plt.pie(x=fraces, labels=labels, autopct='%0.1f%%', explode=explode,
            textprops={'fontsize': 9, 'color': 'black'})
    plt.title('病理类型分布')
    # plt.show()
    plt.savefig(os.path.join(path, 'classes.png'), dpi=100, bbox_inches = 'tight')
    plt.close()

def TIME(path,data):
    #随访时间
    time_dict = data['随访时间'].value_counts().to_dict()
    x = time_dict.keys()
    y = time_dict.values()
    plt.bar(x, y)
    plt.title('随访时间分布')
    plt.xlabel('时间/月')
    plt.ylabel('人数')
    # plt.show()
    plt.savefig(os.path.join(path, 'time.png'), dpi=100, bbox_inches = 'tight')
    plt.close()

def RESULT(path,data):
    # 是否存活
    N = {'N-无瘤生存': 0, 'N-带瘤生存': 0, 'N-复发死亡': 0, 'N-转移死亡': 0, 'N-其他原因死亡': 0}
    for i in data['生存状态']:
        if i == 10:
            N['N-无瘤生存'] += 1
        if i == 11:
            N['N-带瘤生存'] += 1
        if i == 20:
            N['N-复发死亡'] += 1
        if i == 21:
            N['N-转移死亡'] += 1
        if i == 22:
            N['N-其他原因死亡'] += 1

    N_labels = N.keys()
    N_fraces = N.values()
    explode = [0.1, 0.1, 0, 0, 0]
    plt.pie(x=N_fraces, labels=N_labels, autopct='%0.1f%%', explode=explode,
            textprops={'fontsize': 8, 'color': 'black'})
    plt.title('生存状态')
    # plt.show()
    plt.savefig(os.path.join(path, 'result.png'), dpi=100, bbox_inches = 'tight')
    plt.close()

'''#######################################################################################################'''

def onehot_normalize(data):
    #针对无复发时间数据，对数据进行独热编码和标准化
    # 性别
    sex = pd.get_dummies(data['性别'], prefix='性别')
    sex.columns = ['性别-男', '性别-女']
    # 年龄
    age_max = data['年龄'].max()
    age_min = data['年龄'].min()
    age = data['年龄'].apply(lambda x: (x - age_min) / (age_max-age_min))
    age = pd.DataFrame(age, columns=['年龄'])
    # 发病部位
    area = pd.get_dummies(data['发病部位'], prefix='发病部位')
    area.columns = ['发病部位-腮腺', '发病部位-颌下腺', '发病部位-舌下腺+口底', '发病部位-颚', '发病部位-磨牙后区',
                    '发病部位-颊', '发病部位-舌', '发病部位-唇', '发病部位-上颌', '发病部位-其他']
    # 病理类型
    classes = pd.DataFrame(data['病理类型'], columns=['病理类型'])
    classes.loc[classes['病理类型'] == '1a'] = 1
    classes.loc[classes['病理类型'] == '1a '] = 1
    classes.loc[classes['病理类型'] == '1b'] = 1
    classes.loc[classes['病理类型'] == '1c'] = 1
    classes = pd.get_dummies(classes['病理类型'], prefix='病理类型')
    classes.columns = ['病理类型-分化粘表', '病理类型-腺样囊性癌', '病理类型-癌在多形性腺瘤中', '病理类型-非特异性腺癌',
                       '病理类型-腺泡细胞癌', '病理类型-肌上皮癌', '病理类型-多型性腺癌', '病理类型-基底细胞腺癌',
                       '病理类型-唾液腺导管癌', '病理类型-鳞状细胞癌', '病理类型-淋巴上皮癌', '病理类型-(上皮)肌上皮癌',
                       '病理类型-嗜酸细胞腺癌', '病理类型-透明细胞癌', '病理类型-其他']
    # T-分期
    T = data['T分期'].apply(lambda x: (x - 1) / 3)
    T = pd.DataFrame(T, columns=['T分期'])
    # N-分期
    N = data['N分期'].apply(lambda x: int(x) / 2)
    N = pd.DataFrame(N, columns=['N分期'])
    # 局部复发
    local = pd.get_dummies(data['局部复发'], prefix='局部复发')
    local.columns = ['局部复发-无', '局部复发-有']
    # 颈部复发
    neck = pd.get_dummies(data['颈部复发'], prefix='颈部复发')
    neck.columns = ['颈部复发-无', '颈部复发-有']
    # 远处转移
    transform = pd.get_dummies(data['远处转移'], prefix='远处转移')
    transform.columns = ['远处转移-术前无、术后无', '远处转移-术前无、术后有', '远处转移-术前有']
    # 放疗或粒子
    radiotherapy = pd.get_dummies(data['放疗或粒子'], prefix='放疗或粒子')
    radiotherapy.columns = ['放疗或粒子-无', '放疗或粒子-有']
    # 化疗
    chemotherapy = pd.get_dummies(data['化疗'], prefix='化疗')
    chemotherapy.columns = ['化疗-无', '化疗-有']

    data_feature = pd.concat([sex, age, area, classes, T, N, local, neck,  transform, radiotherapy, chemotherapy],
                             axis=1, ignore_index=False)
    data_label = pd.DataFrame(data['全因死亡'],columns=['全因死亡'])

    data = pd.concat([data_feature, data_label], axis=1)

    return data_feature, data_label, data


def fun(x):
    if x <= 24:
        return 0
    if 24 < x <= 60:
        return 1
    else:
        return 2


def onehot_normalize_predictive(data):
    #针对带复发时间数据，对数据进行独热编码和标准化
    # 性别
    sex = pd.get_dummies(data['性别'], prefix='性别')
    sex.columns = ['性别-男', '性别-女']
    # 年龄
    age_max = data['年龄'].max()
    age_min = data['年龄'].min()
    age = data['年龄'].apply(lambda x: (x - age_min) / (age_max-age_min))
    age = pd.DataFrame(age, columns=['年龄'])
    # 发病部位
    area = pd.get_dummies(data['发病部位'], prefix='发病部位')
    area.columns = ['发病部位-腮腺', '发病部位-颌下腺', '发病部位-舌下腺+口底', '发病部位-颚', '发病部位-磨牙后区',
                    '发病部位-颊', '发病部位-舌', '发病部位-唇', '发病部位-上颌', '发病部位-其他']
    # 病理类型
    classes = pd.DataFrame(data['病理类型'], columns=['病理类型'])
    classes.loc[classes['病理类型'] == '1a'] = 1
    classes.loc[classes['病理类型'] == '1a '] = 1
    classes.loc[classes['病理类型'] == '1b'] = 1
    classes.loc[classes['病理类型'] == '1c'] = 1
    classes = pd.get_dummies(classes['病理类型'], prefix='病理类型')
    classes.columns = ['病理类型-分化粘表', '病理类型-腺样囊性癌', '病理类型-癌在多形性腺瘤中', '病理类型-非特异性腺癌',
                       '病理类型-腺泡细胞癌', '病理类型-肌上皮癌', '病理类型-基底细胞腺癌',
                       '病理类型-唾液腺导管癌', '病理类型-鳞状细胞癌', '病理类型-淋巴上皮癌', '病理类型-(上皮)肌上皮癌',
                       '病理类型-嗜酸细胞腺癌', '病理类型-透明细胞癌', '病理类型-其他']
    # T-分期
    T = data['T分期'].apply(lambda x: (x - 1) / 3)
    T = pd.DataFrame(T, columns=['T分期'])
    # N-分期
    N = data['N分期'].apply(lambda x: int(x) / 2)
    N = pd.DataFrame(N, columns=['N分期'])
    # M-分期
    M = data['M分期'].apply(lambda x: int(x))
    M = pd.DataFrame(M, columns=['M分期'])

    data_feature = pd.concat([sex, age, area, classes, T, N, M],
                             axis=1, ignore_index=False)

    data_label = pd.DataFrame(data['随访时间'],columns=['随访时间'])

    data = pd.concat([data_feature, data_label], axis=1)

    data = data.sort_values(by='随访时间')
    data = data.reset_index(drop=True)
    data['随访时间'] = data['随访时间'].apply(lambda x: fun(x))

    return data





'''############################################################################################################'''
#特征重要度排序

def sklearn_select(data_feature, data_label, K, method, path):
    #根据卡方检验选取特征，其中K为选取前多少特征
    X = data_feature
    Y = data_label
    test = SelectKBest(score_func=method, k=K)
    fit = test.fit(X, Y)
    numpy.set_printoptions(precision=1)
    result = pd.DataFrame(fit.scores_, index=list(data_feature.columns), columns=['分数'])
    result.sort_values("分数", ascending=False, inplace=True)
    score = list(result['分数'])
    name = result._stat_axis.values.tolist()
    plt.barh(name, score, height=0.7, color='steelblue', alpha=0.8, )  # 从下往上画
    plt.yticks(fontsize=5)
    plt.xlabel("重要度")
    if method==chi2:
        plt.title("特征重要度排序/卡方检验")
        for x, y in enumerate(score):
            plt.text(y + 0.2, x - 0.1, '%s' % round(y,2),size=5)
        # plt.show()
        plt.savefig(os.path.join(path, 'chi2.png'), dpi=210, bbox_inches = 'tight')
        plt.close()
    if method==f_classif:
        plt.title("特征重要度排序/F检验")
        for x, y in enumerate(score):
            plt.text(y + 0.2, x - 0.1, '%s' % round(y,2),size=5)
        # plt.show()
        plt.savefig(os.path.join(path, 'F.png'), dpi=210, bbox_inches = 'tight')
        plt.close()
    if method==mutual_info_classif:
        plt.title("特征重要度排序/信息增益")
        for x, y in enumerate(score):
            plt.text(y + 0.001, x, '%s' % round(y,3),size=5)
        # plt.show()
        plt.savefig(os.path.join(path, 'inforn.png'), dpi=210, bbox_inches = 'tight')
        plt.close()
    result = result.iloc[:K, :]
    return result


def sns_heatmap(data_feature, data_label, method, path):
    # 利用corr函数计算相关性，包括{‘pearson’, ‘kendall’, ‘spearman’}，然后按照热力图和条形图展示

    data = pd.concat([data_feature, data_label], axis=1, ignore_index=False)
    data_corr = data.corr(method=method) #计算相关性

    #热力图展示
    sns.heatmap(data_corr, linewidths = 0.05, annot=True, square=True,xticklabels=True,yticklabels=True,
                cmap='rainbow',fmt='.2f',annot_kws={'size':2.5,'weight':'bold'})
    plt.xticks(fontsize=5)  # 将字体进行旋转
    plt.yticks(fontsize=5)
    plt.title('特征分布热力图/%s'%method)
    plt.savefig(os.path.join(path, 'corr_%s'%method), dpi=200, bbox_inches='tight')
    # plt.show()
    plt.close()

    # #选出特征与“全因死亡”相关性的一列，以条形图展示
    # result = pd.DataFrame(data_corr['全因死亡'])
    # result.drop(['全因死亡'], axis=0, inplace=True)
    # result.sort_values("全因死亡", ascending=False, inplace=True)
    # score = list(result['全因死亡'])
    # name = result._stat_axis.values.tolist()
    # plt.barh(name, score, height=0.7, color='steelblue', alpha=0.8, )  # 从下往上画
    # plt.yticks(fontsize=5)
    # plt.xlabel("重要度")
    # plt.title("特征重要度排序/%s"%method)
    # for x, y in enumerate(score):
    #     plt.text(y + 0.001, x, '%s' % round(y, 4), size=8)
    # plt.savefig(os.path.join(path, '%s' % method), dpi=210, bbox_inches='tight')
    # # plt.show()
    # plt.close()


def tree(data_feature, data_label, path):
    #决策树取特征
    model = DecisionTreeClassifier()
    model.fit(data_feature, data_label)
    # 打印出每个特征的重要性
    importance = pd.DataFrame()
    importance['name'] = data_feature.columns
    importance['value'] = model.feature_importances_
    importance.sort_values('value', ascending=False, inplace=True)

    plt.bar(importance['name'], importance['value'], 0.4)
    plt.xlabel("特征")
    plt.xticks(rotation=90, fontsize=6)
    plt.ylabel("重要性")
    plt.title("特征重要度排序/决策树")
    for a, b in zip(importance['name'], importance['value']):
        plt.text(a, b + 0.001, '%.4f' % b, ha='center', va='bottom', fontsize=3)
    plt.savefig(os.path.join(path, 'desicionTree.png'), dpi=210, bbox_inches='tight')
    # plt.show()
    plt.close()


def feature_chi(data_feature, data_label, path):
    # 根据卡方检验选取特征，其中K为选取前多少特征
    X = data_feature
    Y = data_label
    test = SelectKBest(score_func=chi2, k=41)
    fit = test.fit(X, Y)
    numpy.set_printoptions(precision=1)
    result = pd.DataFrame(fit.scores_, index=list(data_feature.columns), columns=['分数'])
    result.sort_values("分数", ascending=False, inplace=True)

    score = list(result['分数'])
    name = result._stat_axis.values.tolist()
    plt.barh(name, score, height=0.7, color='steelblue', alpha=0.8, )  # 从下往上画
    plt.yticks(fontsize=5)
    plt.xlabel("重要度")
    plt.title("特征重要度排序/卡方检验")
    for x, y in enumerate(score):
        plt.text(y + 0.2, x - 0.1, '%s' % round(y, 2), size=5)
    # plt.show()
    plt.savefig(os.path.join(path, 'chi2.png'), dpi=210, bbox_inches='tight')
    plt.close()

    result_name = pd.DataFrame(name, columns=['卡方检验'])
    return result_name

def feature_f(data_feature, data_label, path):
    # 根据F检验选取特征，其中K为选取前多少特征
    X = data_feature
    Y = data_label
    test = SelectKBest(score_func=f_classif, k=41)
    fit = test.fit(X, Y)
    numpy.set_printoptions(precision=1)
    result = pd.DataFrame(fit.scores_, index=list(data_feature.columns), columns=['分数'])
    result.sort_values("分数", ascending=False, inplace=True)
    score = list(result['分数'])
    name = result._stat_axis.values.tolist()
    plt.barh(name, score, height=0.7, color='steelblue', alpha=0.8, )  # 从下往上画
    plt.yticks(fontsize=5)
    plt.xlabel("重要度")
    plt.title("特征重要度排序/F检验")
    for x, y in enumerate(score):
        plt.text(y + 0.2, x - 0.1, '%s' % round(y, 2), size=5)
    # plt.show()
    plt.savefig(os.path.join(path, 'F.png'), dpi=210, bbox_inches='tight')
    plt.close()

    result_name = pd.DataFrame(name, columns=['F检验'])
    return result_name

def feature_info(data_feature, data_label, path):
    # 根据信息增益选取特征，其中K为选取前多少特征
    X = data_feature
    Y = data_label
    test = SelectKBest(score_func=mutual_info_classif, k=41)
    fit = test.fit(X, Y)
    numpy.set_printoptions(precision=1)
    result = pd.DataFrame(fit.scores_, index=list(data_feature.columns), columns=['分数'])
    result.sort_values("分数", ascending=False, inplace=True)
    score = list(result['分数'])
    name = result._stat_axis.values.tolist()
    plt.barh(name, score, height=0.7, color='steelblue', alpha=0.8, )  # 从下往上画
    plt.yticks(fontsize=5)
    plt.xlabel("重要度")
    plt.title("特征重要度排序/信息增益")
    for x, y in enumerate(score):
        plt.text(y + 0.001, x, '%s' % round(y, 3), size=5)
    # plt.show()
    plt.savefig(os.path.join(path, 'inforn.png'), dpi=210, bbox_inches='tight')
    plt.close()
    result_name = pd.DataFrame(name, columns=['信息增益'])
    return result_name

def feature_pearson(data_feature, data_label, path):
    # 利用corr函数计算相关性，包括{‘pearson’, ‘kendall’, ‘spearman’}，然后按照热力图和条形图展示
    data = pd.concat([data_feature, data_label], axis=1, ignore_index=False)
    data_corr = data.corr(method='pearson') #计算相关性
    #选出特征与“全因死亡”相关性的一列
    result = pd.DataFrame(data_corr['全因死亡'])
    result.drop(['全因死亡'], axis=0, inplace=True)
    result.sort_values("全因死亡", ascending=False, inplace=True)

    score = list(result['全因死亡'])
    name = result._stat_axis.values.tolist()
    plt.barh(name, score, height=0.7, color='steelblue', alpha=0.8, )  # 从下往上画
    plt.yticks(fontsize=5)
    plt.xlabel("重要度")
    plt.title("特征重要度排序/pearsom")
    for x, y in enumerate(score):
        plt.text(y + 0.001, x, '%s' % round(y, 4), size=8)
    plt.savefig(os.path.join(path, 'pearson.png'), dpi=210, bbox_inches='tight')
    # plt.show()
    plt.close()

    result_name = pd.DataFrame(name, columns=['Pearson'])
    return result_name

def feature_spearman(data_feature, data_label, path):
    # 利用corr函数计算相关性，包括{‘pearson’, ‘kendall’, ‘spearman’}，然后按照热力图和条形图展示
    data = pd.concat([data_feature, data_label], axis=1, ignore_index=False)
    data_corr = data.corr(method='spearman') #计算相关性
    #选出特征与“全因死亡”相关性的一列
    result = pd.DataFrame(data_corr['全因死亡'])
    result.drop(['全因死亡'], axis=0, inplace=True)
    result.sort_values("全因死亡", ascending=False, inplace=True)

    score = list(result['全因死亡'])
    name = result._stat_axis.values.tolist()
    plt.barh(name, score, height=0.7, color='steelblue', alpha=0.8, )  # 从下往上画
    plt.yticks(fontsize=5)
    plt.xlabel("重要度")
    plt.title("特征重要度排序/spearman")
    for x, y in enumerate(score):
        plt.text(y + 0.001, x, '%s' % round(y, 4), size=8)
    plt.savefig(os.path.join(path, 'spearman.png'), dpi=210, bbox_inches='tight')
    # plt.show()
    plt.close()

    result_name = pd.DataFrame(name, columns=['Spearman'])
    return result_name

def feature_tree(data_feature, data_label, path):
    #决策树取特征
    model = DecisionTreeClassifier()
    model.fit(data_feature, data_label)
    # 打印出每个特征的重要性
    importance = pd.DataFrame()
    importance['决策树'] = data_feature.columns
    importance['value'] = model.feature_importances_
    importance.sort_values('value', ascending=False, inplace=True)

    plt.bar(importance['决策树'], importance['value'], 0.4)
    plt.xlabel("特征")
    plt.xticks(rotation=90, fontsize=6)
    plt.ylabel("重要性")
    plt.title("特征重要度排序/决策树")
    for a, b in zip(importance['决策树'], importance['value']):
        plt.text(a, b + 0.001, '%.4f' % b, ha='center', va='bottom', fontsize=3)
    plt.savefig(os.path.join(path, 'desicionTree.png'), dpi=210, bbox_inches='tight')
    # plt.show()
    plt.close()


    # result_name = pd.DataFrame(importance['name'], columns=['决策树'])
    importance.drop('value', axis=1, inplace=True)
    return importance

'''##############################################################################################'''
#模型训练
def lightgbm_train(x_train, y_train):
    '''模型训练'''
    seed = 7

    ## 网格搜索最优参数组合
    learning_rate = [0.001, 0.005, 0.01, 0.015, 0.02]
    num_leaves = [5, 10, 15, 20, 25]

    parameters = {'learning_rate': learning_rate,
                  'num_leaves': num_leaves,

                  }
    kflod = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)
    model = LGBMClassifier(is_unbalance=True)
    clf = GridSearchCV(model, parameters, cv=kflod, scoring='accuracy', verbose=3, n_jobs=-1)
    clf = clf.fit(x_train, y_train)

    ## print每一种组合的结果
    print("分类：最优参数组合：%s" % (clf.best_params_))
    # means = clf.cv_results_['mean_test_score']
    # params = clf.cv_results_['params']
    # for mean, param in zip(means, params):
    #     print("%f  with:   %r" % (mean, param))


    ##训练模型
    clf = LGBMClassifier(learning_rate = clf.best_params_['learning_rate'],
                        num_leaves = clf.best_params_['num_leaves'],
                         is_unbalance=True)
    clf.fit(x_train, y_train)

    #模型存储
    if not os.path.exists('./data/model'):
        os.mkdir('./data/model')
    joblib.dump(clf, os.path.join('./data/model','lightgbm_model.pkl'))
    print('分类：分类模型已保存！')


def lightgbm_train_predictive(x_train, y_train):
    '''模型训练'''
    seed = 7

    ## 网格搜索最优参数组合
    learning_rate = [0.001, 0.005, 0.01, 0.015, 0.02]
    num_leaves = [5, 10, 15, 20, 25]

    parameters = {'learning_rate': learning_rate,
                  'num_leaves': num_leaves,

                  }
    kflod = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)
    model = LGBMClassifier(is_unbalance=True)
    clf = GridSearchCV(model, parameters, cv=kflod, scoring='accuracy', verbose=3, n_jobs=-1)
    clf = clf.fit(x_train, y_train)

    ## print每一种组合的结果
    print("预测：最优参数组合：%s" % (clf.best_params_))
    # means = clf.cv_results_['mean_test_score']
    # params = clf.cv_results_['params']
    # for mean, param in zip(means, params):
    #     print("%f  with:   %r" % (mean, param))


    ##训练模型
    clf = LGBMClassifier(learning_rate = clf.best_params_['learning_rate'],
                        num_leaves = clf.best_params_['num_leaves'],
                         is_unbalance=True)
    clf.fit(x_train, y_train)

    #模型存储
    if not os.path.exists('./data/model'):
        os.mkdir('./data/model')
    joblib.dump(clf, os.path.join('./data/model','lightgbm_model_pre.pkl'))
    print('预测：预测模型已保存！')

'''##############################################################################################'''
#患者模型预测

def model_test_classification(sex_example, age_example, area_example, classes_example,
                              T_example, N_example, M_example, time_example, local_example,
                              neck_example, transform_example, radiotherapy_example, chemotherapy_example):
    sex = {'男': [1, 0], '女': [0, 1]}  # 男，女

    area = {'腮腺': [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], '颌下腺': [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            '舌下腺+口底': [0, 0, 1, 0, 0, 0, 0, 0, 0, 0], '颚': [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            '磨牙后区': [0, 0, 0, 0, 1, 0, 0, 0, 0, 0], '颊': [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            '舌': [0, 0, 0, 0, 0, 0, 1, 0, 0, 0], '唇': [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            '上颌': [0, 0, 0, 0, 0, 0, 0, 0, 1, 0], '其他': [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]}

    classes = {'分化粘表': [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               '腺样囊性癌': [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               '癌在多形性腺瘤中': [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               '非特异性腺癌': [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               '腺泡细胞癌': [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               '肌上皮癌': [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               '多型性腺癌': [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
               '基底细胞腺癌': [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
               '唾液腺导管癌': [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
               '鳞状细胞癌': [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
               '淋巴上皮癌': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
               '上皮-肌上皮癌': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
               '嗜酸细胞腺癌': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
               '透明细胞癌': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
               '其他': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]}

    local = {'局部无复发': [1, 0], '局部有复发': [0, 1]}

    neck = {'颈部无复发': [1, 0], '颈部有复发': [0, 1]}

    transform = {'术前无转移、术后无转移': [1, 0, 0], '术前无转移、术后有转移': [0, 1, 0], '术前有转移': [0, 0, 1]}

    radiotherapy = {'否': [1, 0], '是': [0, 1], '未知':[0,0]}

    chemotherapy = {'否': [1, 0], '是': [0, 1], '未知':[0,0]}

    if sex_example == '男':
        SEX = sex['男']
    if sex_example == '女':
        SEX = sex['女']

    data_age = pd.read_excel(os.path.join('./data/raw_data', '1672例唾液腺恶性肿瘤.xlsx'), usecols=[1])
    AGE = (age_example - data_age.loc[:, '年龄'].min()) / (data_age.loc[:, '年龄'].max() - data_age.loc[:, '年龄'].min())

    if area_example == '腮腺':
        AREA = area['腮腺']
    if area_example == '颌下腺':
        AREA = area['颌下腺']
    if area_example == '舌下腺+口底':
        AREA = area['舌下腺+口底']
    if area_example == '颚':
        AREA = area['颚']
    if area_example == '磨牙后区':
        AREA = area['磨牙后区']
    if area_example == '颊':
        AREA = area['颊']
    if area_example == '舌':
        AREA = area['舌']
    if area_example == '唇':
        AREA = area['唇']
    if area_example == '上颌':
        AREA = area['上颌']
    if area_example == '其他':
        AREA = area['其他']

    if classes_example == '高分化粘表' or classes_example == '中分化粘表' or classes_example == '低分化粘表':
        CLASSES = classes['分化粘表']
    if classes_example == '腺样囊性癌':
        CLASSES = classes['腺样囊性癌']
    if classes_example == '癌在多形性腺瘤中':
        CLASSES = classes['癌在多形性腺瘤中']
    if classes_example == '非特异性腺癌':
        CLASSES = classes['非特异性腺癌']
    if classes_example == '腺泡细胞癌':
        CLASSES = classes['腺泡细胞癌']
    if classes_example == '肌上皮癌':
        CLASSES = classes['肌上皮癌']
    if classes_example == '多型性腺癌':
        CLASSES = classes['多型性腺癌']
    if classes_example == '基底细胞腺癌':
        CLASSES = classes['基底细胞腺癌']
    if classes_example == '唾液腺导管癌':
        CLASSES = classes['唾液腺导管癌']
    if classes_example == '鳞状细胞癌':
        CLASSES = classes['鳞状细胞癌']
    if classes_example == '淋巴上皮癌':
        CLASSES = classes['淋巴上皮癌']
    if classes_example == '上皮-肌上皮癌':
        CLASSES = classes['上皮-肌上皮癌']
    if classes_example == '嗜酸细胞腺癌':
        CLASSES = classes['嗜酸细胞腺癌']
    if classes_example == '透明细胞癌':
        CLASSES = classes['透明细胞癌']
    if classes_example == '其他':
        CLASSES = classes['其他']

    T = (T_example - 1) / 3

    N = N_example / 3

    if local_example == '否':
        LOCAL = local['局部无复发']
    if local_example != '否':
        LOCAL = local['局部有复发']

    if neck_example == '否':
        NECK = neck['颈部无复发']
    if neck_example != '否':
        NECK = neck['颈部有复发']

    if M_example == 0 and transform_example == '否':
        TRANSFORM = transform['术前无转移、术后无转移']
    if M_example == 0 and transform_example != '否':
        TRANSFORM = transform['术前无转移、术后有转移']
    if M_example == 1:
        TRANSFORM = transform['术前有转移']

    if radiotherapy_example == '否':
        RADIOTHERAPY = radiotherapy['否']
    if radiotherapy_example == '是':
        RADIOTHERAPY = radiotherapy['是']
    if radiotherapy_example == '未知':
        RADIOTHERAPY = radiotherapy['未知']
    if chemotherapy_example == '否':
        CHEMOTHERAPY = chemotherapy['否']
    if chemotherapy_example == '是':
        CHEMOTHERAPY = chemotherapy['是']
    if chemotherapy_example == '未知':
        CHEMOTHERAPY = chemotherapy['未知']

    X = [SEX + [AGE] + AREA + CLASSES + [T, N] + LOCAL + NECK + TRANSFORM + RADIOTHERAPY + CHEMOTHERAPY]

    return X


def model_test_predictive(sex_example, age_example, area_example, classes_example,
                              T_example, N_example, M_example, time_example, local_example,
                              neck_example, transform_example, radiotherapy_example, chemotherapy_example):
    sex = {'男': [1, 0], '女': [0, 1]}  # 男，女

    area = {'腮腺': [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], '颌下腺': [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            '舌下腺+口底': [0, 0, 1, 0, 0, 0, 0, 0, 0, 0], '颚': [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            '磨牙后区': [0, 0, 0, 0, 1, 0, 0, 0, 0, 0], '颊': [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            '舌': [0, 0, 0, 0, 0, 0, 1, 0, 0, 0], '唇': [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            '上颌': [0, 0, 0, 0, 0, 0, 0, 0, 1, 0], '其他': [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]}

    # 注意：由于多形性腺癌在最终死亡的患者中并未出现，所以向量是14维度，而不是15维度，其多形性腺癌对应全零向量
    classes = {'分化粘表': [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], '腺样囊性癌': [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               '癌在多形性腺瘤中': [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               '非特异性腺癌': [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               '腺泡细胞癌': [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], '肌上皮癌': [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
               '多型性腺癌': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               '基底细胞腺癌': [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
               '唾液腺导管癌': [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
               '鳞状细胞癌': [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
               '淋巴上皮癌': [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
               '上皮-肌上皮癌': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
               '嗜酸细胞腺癌': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
               '透明细胞癌': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0], '其他': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]}

    if sex_example == '男':
        SEX = sex['男']
    if sex_example == '女':
        SEX = sex['女']

    data_age = pd.read_excel(os.path.join('./data/raw_data', '1672例唾液腺恶性肿瘤.xlsx'), usecols=[1])
    AGE = (age_example - data_age.loc[:, '年龄'].min()) / (data_age.loc[:, '年龄'].max() - data_age.loc[:, '年龄'].min())

    if area_example == '腮腺':
        AREA = area['腮腺']
    if area_example == '颌下腺':
        AREA = area['颌下腺']
    if area_example == '舌下腺+口底':
        AREA = area['舌下腺+口底']
    if area_example == '颚':
        AREA = area['颚']
    if area_example == '磨牙后区':
        AREA = area['磨牙后区']
    if area_example == '颊':
        AREA = area['颊']
    if area_example == '舌':
        AREA = area['舌']
    if area_example == '唇':
        AREA = area['唇']
    if area_example == '上颌':
        AREA = area['上颌']
    if area_example == '其他':
        AREA = area['其他']

    if classes_example == '高分化粘表' or classes_example == '中分化粘表' or classes_example == '低分化粘表':
        CLASSES = classes['分化粘表']
    if classes_example == '腺样囊性癌':
        CLASSES = classes['腺样囊性癌']
    if classes_example == '癌在多形性腺瘤中':
        CLASSES = classes['癌在多形性腺瘤中']
    if classes_example == '非特异性腺癌':
        CLASSES = classes['非特异性腺癌']
    if classes_example == '腺泡细胞癌':
        CLASSES = classes['腺泡细胞癌']
    if classes_example == '肌上皮癌':
        CLASSES = classes['肌上皮癌']
    if classes_example == '多型性腺癌':
        CLASSES = classes['多型性腺癌']
    if classes_example == '基底细胞腺癌':
        CLASSES = classes['基底细胞腺癌']
    if classes_example == '唾液腺导管癌':
        CLASSES = classes['唾液腺导管癌']
    if classes_example == '鳞状细胞癌':
        CLASSES = classes['鳞状细胞癌']
    if classes_example == '淋巴上皮癌':
        CLASSES = classes['淋巴上皮癌']
    if classes_example == '上皮-肌上皮癌':
        CLASSES = classes['上皮-肌上皮癌']
    if classes_example == '嗜酸细胞腺癌':
        CLASSES = classes['嗜酸细胞腺癌']
    if classes_example == '透明细胞癌':
        CLASSES = classes['透明细胞癌']
    if classes_example == '其他':
        CLASSES = classes['其他']

    T = (T_example - 1) / 3

    N = N_example / 3

    M = M_example

    X = [SEX + [AGE] + AREA + CLASSES + [T, N, M]]

    return X


























