#-*- coding:utf-8 -*-
"""
author:zhangxun
"""
import pandas as pd
import os
from sklearn import tree
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

# 读取数据
path = os.getcwd()
save_path = os.path.join(path,'data','train.csv')
with open(save_path,'r') as trainFile:
    train_data = pd.read_csv(trainFile)
    y = train_data['y']
    X = train_data.drop(columns=['y'])

# 标签转换为0/1
y =y.replace(1,0)
y =y.replace(2,1)

# 拆分训练数据与测试数据
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

# 使用信息熵作为划分标准，对决策树进行训练
clf = tree.DecisionTreeClassifier(criterion='entropy')
clf.fit(x_train, y_train)

# 把决策树结构写入文件 '''
# with open("tree.dot", 'w') as f:
#     f = tree.export_graphviz(clf, out_file=f)

# 预测
predict_y = clf.predict(x_test)
# 准确率与召回率
precision, recall, thresholds = precision_recall_curve(y_test, clf.predict(x_test))
# predict_y = clf.predict_proba(X)[:,1]
print(classification_report(y_test, predict_y, target_names = ['no', 'yes']))