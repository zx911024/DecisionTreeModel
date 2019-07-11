#-*- coding:utf-8 -*-
"""
author:zhangxun
Created on 2019-04-22 15:13
"""
import os
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics

path = os.getcwd()
save_path = os.path.join(path,'data','train.csv')
with open(save_path,'r') as trainFile:
    train_data = pd.read_csv(trainFile)
    y = train_data['y']
    x = train_data.drop(columns=['y'])
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25)
#确认模型
cart=DecisionTreeClassifier(criterion='gini')
'''DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
    max_features=None, max_leaf_nodes=None,            
    min_impurity_split=1e-07, min_samples_leaf=1,            
    min_samples_split=2, min_weight_fraction_leaf=0.0,            
    presort=False, random_state=None, splitter='best')'''

#训练模型
cart.fit(x_train,y_train)
cart.score(x_test,y_test)

#展示模型预测结果
print(metrics.classification_report(y_test,cart.predict(x_test)))
print(metrics.confusion_matrix(y_test,cart.predict(x_test)))
