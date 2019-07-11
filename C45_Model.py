# coding:utf-8
from math import log
import pandas as pd
from sklearn import metrics
import os
import operator
from sklearn.model_selection import train_test_split

# 计算信息熵
def jisuanEnt(dataset):
    numEntries = len(dataset)
    labelCounts = {}
    # 给所有可能分类创建字典
    for featVec in dataset:
        currentlabel = featVec[-1]
        if currentlabel not in labelCounts.keys():
            labelCounts[currentlabel] = 0
        labelCounts[currentlabel] += 1
    Ent = 0.0
    for key in labelCounts:
        p = float(labelCounts[key]) / numEntries
        Ent = Ent - p * log(p, 2)  # 以2为底求对数
    return Ent

# 划分数据集
def splitdataset(dataset, axis, value):
    retdataset = []  # 创建返回的数据集列表
    for featVec in dataset:  # 抽取符合划分特征的值
        if featVec[axis] == value:
            reducedfeatVec = featVec[:axis]  # 去掉axis特征
            reducedfeatVec.extend(featVec[axis + 1:])  # 将符合条件的特征添加到返回的数据集列表
            retdataset.append(reducedfeatVec)
    return retdataset

'''
选择最好的数据集划分方式
C4.5算法：使用“增益率”来选择划分属性
'''
# C4.5算法
def C45_chooseBestFeatureToSplit(dataset):
    numFeatures = len(dataset[0]) - 1
    baseEnt = jisuanEnt(dataset)
    bestInfoGain_ratio = 0.0
    bestFeature = -1
    for i in range(numFeatures):  # 遍历所有特征
        featList = [example[i] for example in dataset]
        uniqueVals = set(featList)  # 将特征列表创建成为set集合，元素不可重复。创建唯一的分类标签列表
        newEnt = 0.0
        IV = 0.0
        for value in uniqueVals:  # 计算每种划分方式的信息熵
            subdataset = splitdataset(dataset, i, value)
            p = len(subdataset) / float(len(dataset))
            newEnt += p * jisuanEnt(subdataset)
            IV = IV - p * log(p, 2)
        infoGain = baseEnt - newEnt
        if (IV == 0):  # fix the overflow bug
            continue
        infoGain_ratio = infoGain / IV  # 这个feature的infoGain_ratio
        # print(u"C4.5中第%d个特征的信息增益率为：%.3f"%(i,infoGain_ratio))
        if (infoGain_ratio > bestInfoGain_ratio):  # 选择最大的gain ratio
            bestInfoGain_ratio = infoGain_ratio
            bestFeature = i  # 选择最大的gain ratio对应的feature
    return bestFeature

def majorityCnt(classList):
    '''
    数据集已经处理了所有属性，但是类标签依然不是唯一的，
    此时我们需要决定如何定义该叶子节点，在这种情况下，我们通常会采用多数表决的方法决定该叶子节点的分类
    '''
    classCont = {}
    for vote in classList:
        if vote not in classCont.keys():
            classCont[vote] = 0
        classCont[vote] += 1
    sortedClassCont = sorted(classCont.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCont[0][0]


def C45_createTree(dataset, labels):
    classList = [example[-1] for example in dataset]
    if classList.count(classList[0]) == len(classList):
        # 类别完全相同，停止划分
        return classList[0]
    if len(dataset[0]) == 1:
        # 遍历完所有特征时返回出现次数最多的
        return majorityCnt(classList)
    bestFeat = C45_chooseBestFeatureToSplit(dataset)
    bestFeatLabel = labels[bestFeat]
    # print(u"此时最优索引为："+str(bestFeatLabel))
    C45Tree = {bestFeatLabel: {}}
    del (labels[bestFeat])
    # 得到列表包括节点所有的属性值
    featValues = [example[bestFeat] for example in dataset]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]
        C45Tree[bestFeatLabel][value] = C45_createTree(splitdataset(dataset, bestFeat, value), subLabels)
    return C45Tree


def classify(inputTree, featLabels, testVec):
    """
    输入：决策树，分类标签，测试数据
    输出：决策结果
    描述：跑决策树
    """
    firstStr = list(inputTree.keys())[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    classLabel = '0'
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':
                classLabel = classify(secondDict[key], featLabels, testVec)
            else:
                classLabel = secondDict[key]
    return classLabel


def classifytest(inputTree, featLabels, testDataSet):
    """
    输入：决策树，分类标签，测试数据集
    输出：决策结果
    描述：跑决策树
    """
    classLabelAll = []
    for testVec in testDataSet:
        classLabelAll.append(classify(inputTree, featLabels, testVec))
    return classLabelAll


if __name__ == '__main__':
    # 数据读入
    path = os.getcwd()
    save_path = os.path.join(path, 'data', 'train.csv')
    with open(save_path, 'r') as trainFile:
        train_data = pd.read_csv(trainFile)
        XX = train_data
        y = train_data['y']
        x = train_data.drop(columns=['y'])

        # 标签转换为0/1
        y = y.replace(1, "0")
        y = y.replace(2, "1")

    # 训练数据与测试数据
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

    # 训练数据集需要带上标签
    x_train["y"] = y_train
    dataset = x_train.values.tolist()
    labels = y_train.values.tolist()

    # 测试数据不需要带标签，转换未List
    noClassTest = x_test.values.tolist()
    print("数据集长度：", len(dataset))
    labels_tmp = labels[:]  # 备份输入
    print("----------训练C4.5模型----------")
    C45desicionTree = C45_createTree(dataset, labels_tmp)

    # 预测测试集
    predict_y = classifytest(C45desicionTree, labels, noClassTest)

    # 将测试结果转换为DataFrame
    pre_y = pd.DataFrame(predict_y)
    print("测试数据混淆矩阵：")
    print(metrics.classification_report(y_test, pre_y))


