

import sys
import os
import pandas as pd
import numpy as np
import missingno as mg
import matplotlib.pyplot as plt
import pandas as pd
import numpy
import argparse
import warnings
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2, f_classif, mutual_info_classif
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
import joblib
from collections import Counter


from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QIcon, QPixmap
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication , QMainWindow, QHeaderView, QMessageBox, QWidget
from PyQt5.QtCore import pyqtSignal , Qt
from PyQt5.QtGui import QStandardItemModel,QStandardItem
from PyQt5 import QtCore
QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)


from GUI_log import Ui_Form_Log
from GUI_account import Ui_Form_Account
from GUI_Main import Ui_Form
from GUI_Child import Ui_Form_Child
from data_utils import lack_processing, SEX, AGE, RADIOTHERAPY, CHEMOTHERAPY, T_TNM, N_TNM, M_TNM, TIME, LOCAL_RELAPSE, NECK_RELAPSE, TRANSFORM, AREA, CLASSES, RESULT, onehot_normalize, \
    sklearn_select, sns_heatmap, tree, feature_chi, feature_f, feature_info, feature_pearson, feature_spearman, feature_tree, lightgbm_train, lack_processing_predictive, \
    onehot_normalize_predictive,lightgbm_train_predictive, model_test_classification, model_test_predictive



plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']




def onehot_normalize(data):
    #对数据进行独热编码和标准化
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
    N = data['N分期'].apply(lambda x: int(x) / 3)
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





class GUI_Create_Account(QWidget,Ui_Form_Account):
    '''
    创建新用户界面
    '''
    def __init__(self, user, path):
        super().__init__()
        self.user = user
        self.path = path
        self.setupUi(self)
        self.initUI()

    def initUI(self):
        self.pushButton.clicked.connect(self.create)

    def create(self):
        # 读入用户ID和Passport
        ID = self.lineEdit.text()
        Passport = self.lineEdit_2.text()
        if ID=='' or Passport=='' :
            QMessageBox.warning(self, "警告", "请输入新用户名或密码！")
        else:
            self.user[ID] = Passport
            df = pd.DataFrame.from_dict(self.user, orient='index', columns=['Passport'])
            df = df.reset_index().rename(columns={'index': 'ID'})
            df.to_csv(self.path, encoding='ISO-8859-1')
            QMessageBox.warning(self, "提示", "新用户创建完成，请退出创建界面重新登陆！")





class GUI_Create_Child(QWidget,Ui_Form_Child):
    '''
    子界面：主界面中用于显示图片
    '''
    def __init__(self, path):
        super().__init__()
        self.path = path
        self.setupUi(self)
        self.initUI()

    def initUI(self):
        matrix_png = QPixmap(self.path)
        self.label.setPixmap(matrix_png)
        self.label.setScaledContents(True)

class Stream(QObject):
    """Redirects console output to text widget."""
    newText = pyqtSignal(str)
    def write(self, text):
        self.newText.emit(str(text))


class GUI_Main(QWidget, Ui_Form):
    '''
    主界面，用于数据库展示、数据分析、模型训练及应用
    '''
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.initUI()
        sys.stdout = Stream(newText=self.onUpdateText)

    def onUpdateText(self, text):
        """Write console output to text widget."""
        cursor = self.textBrowser.textCursor()
        cursor.movePosition(QtGui.QTextCursor.End)
        cursor.insertText(text)
        self.textBrowser.setTextCursor(cursor)
        self.textBrowser.ensureCursorVisible()

    def closeEvent(self, event):
        """Shuts down application on close."""
        # Return stdout to defaults.
        sys.stdout = sys.__stdout__
        super().closeEvent(event)



    def initUI(self):

        self.pushButton_15.clicked.connect(self.model_test)  # page-4 对患者进行真实预测

        self.pushButton_16.clicked.connect(self.new_patient_predict) # page-5 对新患者进行真实预测



    def model_test(self):
        '''page-4 对患者进行真实预测'''
        name_example = str(self.lineEdit_5.text())
        tel_example = str(self.lineEdit_6.text())
        address_example = str(self.lineEdit_7.text())

        sex_example = str(self.comboBox.currentText()) #性别
        age_example = int(self.spinBox_2.value()) #年龄
        area_example = str(self.comboBox_2.currentText()) # 发病部位名称
        classes_example = str(self.comboBox_3.currentText()) #病理类型名称
        T_example = int(self.comboBox_7.currentText())  #1,2,3,4 代表等级
        N_example = int(self.comboBox_8.currentText())  #0,1,2,3代表等级
        M_example = int(self.comboBox_9.currentText())  # 0:术前无转移，1：术前有转移
        time_example = int(self.spinBox_3.value()) #整数

        radiotherapy_example = str(self.comboBox_10.currentText())  # 否、是、未知
        chemotherapy_example = str(self.comboBox_11.currentText())  # 否、是、未知

        if str(self.lineEdit_8.text()) == '' or str(self.lineEdit_11.text()) == '' or str(self.lineEdit_12.text()) == '':
            QMessageBox.warning(self, "提示", "患者信息不全！")
        else:
            if str(self.lineEdit_8.text()) == '否': #'否'、是
                local_example = '否'  # 否、月份
            else:
                local_example = str(self.lineEdit_8.text())

            if str(self.lineEdit_11.text()) == '否': #'否'、是
                neck_example = '否'  # 否、月份
            else:
                neck_example = str(self.lineEdit_11.text())

            if str(self.lineEdit_12.text()) == '否': #'否'、是
                transform_example = '否'  # 否、月份
            else:
                transform_example = str(self.lineEdit_12.text())



            X_classification = model_test_classification(sex_example, age_example, area_example, classes_example,
                              T_example, N_example, M_example, time_example, local_example,
                              neck_example, transform_example, radiotherapy_example, chemotherapy_example)
            algorithm_path = os.path.join('./data/model', 'lightgbm_model.pkl')
            if not os.path.exists(algorithm_path):
                QMessageBox.warning(self, "提示", "请先在上一窗口进行模型训练！")
            else:
                gbm_classification = joblib.load(algorithm_path)
                Y_classification = gbm_classification.predict_proba(X_classification)

                self.lineEdit_13.setText(str(Y_classification[0][0]))

                feature_file = './data/feature'
                if not os.path.exists(feature_file):
                    os.makedirs(feature_file)

                plt.pie([Y_classification[0][0], Y_classification[0][1]],
                        explode=[0, 0.1],
                        labels=['存活', '死亡'],
                        colors=['green', 'red'],
                        autopct='%3.2f%%',  # 数值保留固定小数位
                        shadow=True,  # 无阴影设置

                        )
                plt.title('智能辅助分类结果')
                # plt.show()
                plt.savefig(os.path.join('./data/feature','lightgbm_model_classification.png'), dpi=100, bbox_inches='tight')
                plt.close()

                matrix_png = QPixmap('./data/feature/lightgbm_model_classification.png')
                self.label_40.setPixmap(matrix_png)
                self.label_40.setScaledContents(True)
                QMessageBox.warning(self, "提示", "诊断完成！")

                if Y_classification[0][0]<0.5:

                    X_predictive = model_test_predictive(sex_example, age_example, area_example, classes_example,
                                      T_example, N_example, M_example, time_example, local_example,
                                      neck_example, transform_example, radiotherapy_example, chemotherapy_example)
                    algorithm_path = os.path.join('./data/model/lightgbm_model_pre.pkl')
                    if not os.path.exists(algorithm_path):
                        QMessageBox.warning(self, "提示", "请先在上一窗口进行模型训练！")
                    else:
                        gbm_predictive = joblib.load(algorithm_path)

                        Y_predictive = gbm_predictive.predict_proba(X_predictive)
                        self.lineEdit_14.setText(str(Y_predictive[0][0]+Y_predictive[0][1]))
                        # self.lineEdit_15.setText(str(Y_predictive[0][2]))
                        # self.lineEdit_16.setText(str(Y_predictive[0][2]))

                        plt.pie([Y_predictive[0][0]+Y_predictive[0][1], Y_predictive[0][2]],
                                explode=[0.1, 0],
                                labels=['5年', ''],
                                colors=['red', 'white'],
                                autopct='%3.2f%%',  # 数值保留固定小数位
                                shadow=True,  # 无阴影设置
                                startangle=90,  # 逆时针起始角度设置
                                )
                        plt.title('智能辅助预测结果')
                        # plt.show()
                        plt.savefig(os.path.join('./data/feature/lightgbm_model_predictive.png'), dpi=100, bbox_inches='tight')
                        plt.close()

                        matrix_png = QPixmap('./data/feature/lightgbm_model_predictive.png')
                        self.label_41.setPixmap(matrix_png)
                        self.label_41.setScaledContents(True)
                        QMessageBox.warning(self, "提示", "第二阶段诊断完成！")
                else:
                    #若患者生存概率高于50%，不进行寿命区间预测
                    self.lineEdit_14.setText('')
                    # self.lineEdit_15.setText('')
                    # self.lineEdit_16.setText('')
                    self.label_41.setPixmap(QPixmap(''))

                #保存患者数据
                if not os.path.exists('./data/raw_data/patient.xlsx'):
                    data = {'姓名': [name_example], '性别':[sex_example], '年龄':[age_example], '联系方式':[tel_example], '住址':[address_example],
                            '发病部位':[area_example],
                            '病理类型':[classes_example], 'T分期':[T_example], 'N分期':[N_example], 'M分期':[M_example],
                            '随访时间':[time_example], '放疗或粒子':[radiotherapy_example], '化疗':[chemotherapy_example],
                            '局部复发':[local_example], '颈部复发':[neck_example], '远处转移':[transform_example],
                            '预测生存概率':[str(Y_classification[0][0])]}
                    input_data = pd.DataFrame(data)
                    input_data.to_excel(os.path.join('./data/raw_data/patient.xlsx'), index=False)
                else:
                    past_data = pd.read_excel('./data/raw_data/patient.xlsx', engine='openpyxl')
                    data = {'姓名': [name_example], '性别': [sex_example], '年龄': [age_example], '联系方式': [tel_example],
                            '住址': [address_example],
                            '发病部位': [area_example],
                            '病理类型': [classes_example], 'T分期': [T_example], 'N分期': [N_example], 'M分期': [M_example],
                            '随访时间': [time_example], '放疗或粒子': [radiotherapy_example], '化疗': [chemotherapy_example],
                            '局部复发': [local_example], '颈部复发': [neck_example], '远处转移': [transform_example],
                            '预测生存概率':[str(Y_classification[0][0])]}
                    new_data = pd.DataFrame(data)
                    add_data = past_data.append(new_data, ignore_index=True)
                    add_data.to_excel(os.path.join('./data/raw_data/patient.xlsx'), index=False)



    def new_patient_predict(self):
        # page-5 新患者病情预测
        name_example = str(self.lineEdit_17.text())
        tel_example = str(self.lineEdit_18.text())
        address_example = str(self.lineEdit_19.text())

        sex_example = str(self.comboBox_16.currentText())  # 性别
        age_example = int(self.spinBox_4.value())  # 年龄
        area_example = str(self.comboBox_17.currentText())  # 发病部位名称
        classes_example = str(self.comboBox_19.currentText())  # 病理类型名称
        T_example = int(self.comboBox_18.currentText())  # 1,2,3,4 代表等级
        N_example = int(self.comboBox_12.currentText())  # 0,1,2,3代表等级
        M_example = int(self.comboBox_13.currentText())  # 0:术前无转移，1：术前有转移
        time_example = int(self.spinBox_5.value())  # 整数

        radiotherapy_example = str(self.comboBox_14.currentText())  # 否、是、未知
        chemotherapy_example = str(self.comboBox_15.currentText())  # 否、是、未知

        local_example = str(self.comboBox_20.currentText())
        neck_example = str(self.comboBox_21.currentText())
        transform_example = str(self.comboBox_22.currentText())

        # if str(self.lineEdit_8.text()) == '' or str(self.lineEdit_11.text()) == '' or str(
        #         self.lineEdit_12.text()) == '':
        #     QMessageBox.warning(self, "提示", "患者信息不全！")
        # else:
        #     if str(self.lineEdit_8.text()) == '否':  # '否'、是
        #         local_example = '否'  # 否、月份
        #     else:
        #         local_example = str(self.lineEdit_8.text())
        #
        #     if str(self.lineEdit_11.text()) == '否':  # '否'、是
        #         neck_example = '否'  # 否、月份
        #     else:
        #         neck_example = str(self.lineEdit_11.text())
        #
        #     if str(self.lineEdit_12.text()) == '否':  # '否'、是
        #         transform_example = '否'  # 否、月份
        #     else:
        #         transform_example = str(self.lineEdit_12.text())

        X_classification = model_test_classification(sex_example, age_example, area_example, classes_example,
                                                     T_example, N_example, M_example, time_example, local_example,
                                                     neck_example, transform_example, radiotherapy_example,
                                                     chemotherapy_example)
        algorithm_path = os.path.join('./data/model', 'lightgbm_model.pkl')
        if not os.path.exists(algorithm_path):
            QMessageBox.warning(self, "提示", "模型不存在！")
        else:
            gbm_classification = joblib.load(algorithm_path)
            Y_classification = gbm_classification.predict_proba(X_classification)

            self.lineEdit_21.setText(str(Y_classification[0][0]))

            feature_file = './data/feature'
            if not os.path.exists(feature_file):
                os.makedirs(feature_file)

            plt.pie([Y_classification[0][0], Y_classification[0][1]],
                    explode=[0, 0.1],
                    labels=['存活', '死亡'],
                    colors=['green', 'red'],
                    autopct='%3.2f%%',  # 数值保留固定小数位
                    shadow=True,  # 无阴影设置

                    )
            plt.title('智能辅助分类结果')
            # plt.show()
            plt.savefig(os.path.join('./data/feature', 'new_lightgbm_model_classification.png'), dpi=100,
                        bbox_inches='tight')
            plt.close()

            matrix_png = QPixmap('./data/feature/new_lightgbm_model_classification.png')
            self.label_63.setPixmap(matrix_png)
            self.label_63.setScaledContents(True)
            QMessageBox.warning(self, "提示", "诊断完成！")

            if Y_classification[0][0] < 0.5:

                X_predictive = model_test_predictive(sex_example, age_example, area_example, classes_example,
                                                     T_example, N_example, M_example, time_example, local_example,
                                                     neck_example, transform_example, radiotherapy_example,
                                                     chemotherapy_example)
                algorithm_path = os.path.join('./data/model/lightgbm_model_pre.pkl')
                if not os.path.exists(algorithm_path):
                    QMessageBox.warning(self, "提示", "模型不存在！")
                else:
                    gbm_predictive = joblib.load(algorithm_path)

                    Y_predictive = gbm_predictive.predict_proba(X_predictive)
                    self.lineEdit_22.setText(str(Y_predictive[0][0] + Y_predictive[0][1]))
                    # self.lineEdit_15.setText(str(Y_predictive[0][2]))
                    # self.lineEdit_16.setText(str(Y_predictive[0][2]))

                    plt.pie([Y_predictive[0][0] + Y_predictive[0][1], Y_predictive[0][2]],
                            explode=[0.1, 0],
                            labels=['5年', ''],
                            colors=['red', 'white'],
                            autopct='%3.2f%%',  # 数值保留固定小数位
                            shadow=True,  # 无阴影设置
                            startangle=90,  # 逆时针起始角度设置
                            )
                    plt.title('智能辅助预测结果')
                    # plt.show()
                    plt.savefig(os.path.join('./data/feature/new_lightgbm_model_predictive.png'), dpi=100,
                                bbox_inches='tight')
                    plt.close()

                    matrix_png = QPixmap('./data/feature/new_lightgbm_model_predictive.png')
                    self.label_67.setPixmap(matrix_png)
                    self.label_67.setScaledContents(True)
                    QMessageBox.warning(self, "提示", "第二阶段诊断完成！")
            else:
                # 若患者生存概率高于50%，不进行寿命区间预测
                self.lineEdit_22.setText('')
                # self.lineEdit_15.setText('')
                # self.lineEdit_16.setText('')
                self.label_67.setPixmap(QPixmap(''))

            # 保存患者数据
            if not os.path.exists('./data/raw_data/new_patient.xlsx'):
                data = {'姓名': [name_example], '性别': [sex_example], '年龄': [age_example], '联系方式': [tel_example],
                        '住址': [address_example],
                        '发病部位': [area_example],
                        '病理类型': [classes_example], 'T分期': [T_example], 'N分期': [N_example], 'M分期': [M_example],
                        '随访时间': [time_example], '放疗或粒子': [radiotherapy_example], '化疗': [chemotherapy_example],
                        '局部复发': [local_example], '颈部复发': [neck_example], '远处转移': [transform_example],
                        '预测生存概率': [str(Y_classification[0][0])]}
                input_data = pd.DataFrame(data)
                input_data.to_excel(os.path.join('./data/raw_data/new_patient.xlsx'), index=False)
            else:
                past_data = pd.read_excel('./data/raw_data/new_patient.xlsx', engine='openpyxl')
                data = {'姓名': [name_example], '性别': [sex_example], '年龄': [age_example], '联系方式': [tel_example],
                        '住址': [address_example],
                        '发病部位': [area_example],
                        '病理类型': [classes_example], 'T分期': [T_example], 'N分期': [N_example], 'M分期': [M_example],
                        '随访时间': [time_example], '放疗或粒子': [radiotherapy_example], '化疗': [chemotherapy_example],
                        '局部复发': [local_example], '颈部复发': [neck_example], '远处转移': [transform_example],
                        '预测生存概率': [str(Y_classification[0][0])]}
                new_data = pd.DataFrame(data)
                add_data = past_data.append(new_data, ignore_index=True)
                add_data.to_excel(os.path.join('./data/raw_data/new_patient.xlsx'), index=False)


class GUI_Log(QMainWindow, Ui_Form_Log):
    '''
    主页：登录界面
    '''
    def __init__(self, parent=None):
        super(GUI_Log, self).__init__(parent)
        self.setupUi(self)
        self.initUI()

        # 读入用户ID和Passport
        self.path = './data/user_info.csv'
        if os.path.exists(self.path):
            user_info = pd.read_csv(self.path, index_col=False, encoding='ISO-8859-1')
            user_dict = dict(zip(user_info['ID'], user_info['Passport']))
        else:
            user_dict = {}
        self.user = user_dict


    def initUI(self):
        self.pushButton.clicked.connect(self.log_func)
        self.pushButton_2.clicked.connect(self.create_account)


    def log_func(self):
        #获取输入框文本
        account = self.lineEdit_3.text()
        password = self.lineEdit_2.text()
        if account not in self.user.keys() :
            QMessageBox.warning(self, "警告", "不存在此用户，请'重新输入'或'创建新用户'！")
        else:
            if str(self.user[account])==password:
                main_win.show()  #跳转到主窗口
                self.close()    #关闭登录窗口
            else:
                QMessageBox.warning(self, "警告", "密码错误，请'重新输入'！")


    def create_account(self):
        self.create_account_form = GUI_Create_Account(self.user,self.path)
        self.create_account_form.show() #新建用户界面



if __name__=="__main__":

    app = QApplication(sys.argv)
    log_form = GUI_Log() #登录界面
    log_form.show()
    main_win = GUI_Main()  # 主界面
    sys.exit(app.exec_())