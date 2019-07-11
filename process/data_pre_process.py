#-*- coding:utf-8 -*-
"""
author:zhangxun
work:数据预处理
"""
import os
import pandas as pd
from logConf.logger import get_logger
logger = get_logger()

# 读取某列的种类
def col_sample_list(col_one):
    col_list = list(set(col_one))
    return col_list

if __name__ == "__main__":
    path = os.path.pardir
    data_path = os.path.join(path, "data", "bankData.csv")
    save_path = os.path.join(path,'data','train.csv')
    # 读取整个csv文件
    csv_data = pd.read_csv(data_path)
    columns_names = csv_data.columns.values.tolist()
    for i in columns_names:
        if i in ["age",'balance','day','duration','campaign','pdays','previous']:
            continue
        col_one = csv_data[i]
        col_list = col_sample_list(col_one)
        col_lenth = len(col_list)
        for index,j in enumerate(col_list):
            csv_data[i]=csv_data[i].replace(j,index+1)
    csv_data.to_csv(save_path,index=0)

