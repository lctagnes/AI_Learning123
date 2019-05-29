# encoding: utf-8
from math import log
import operator
import treePlotter


# 获取数据
def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return dataSet, labels


dataSet, labels = createDataSet()



# 计算系统熵
def calcShannonEnt(dataSet):
    # 获取数组的长度numEntries
    numEntries = len(dataSet)  
    # 定义一个字典计数类别labelCounts
    labelCounts = {}
    # 计算数据集中不同类别的个数
    for featVec in dataSet:  
         # 获取类别currentLabel
        currentLabel = featVec[-1] 
        # 如果该类别不在字典的key里面，该key的值默认为0次（第一次进来字典是空的）
        if currentLabel not in labelCounts.keys(): labelCounts[currentLabel] = 0
        # 不然就计数加1
        labelCounts[currentLabel] += 1
    # 初始化香农熵shannonEnt
    shannonEnt = 0.0
    # 遍历字典里的key
    for key in labelCounts:
        # 概率
        prob = float(labelCounts[key]) / numEntries  
        # 熵
        shannonEnt -= prob * log(prob, 2)  
    return shannonEnt


shannonEnt = calcShannonEnt(dataSet)

# axis：样本特征的索引0和1  value：去重后的特征值 0和1
def splitDataSet(dataSet, axis, value):
    retDataSet = []
    # 遍历数据集的每个样本
    for featVec in dataSet:
        # 如果每个样本的特定位置的值是给定的去重后的特征值
        if featVec[axis] == value:
            # 前面axis个元素
            reducedFeatVec = featVec[:axis]
            # 从axis + 1个开始
            reducedFeatVec.extend(featVec[axis + 1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet


def chooseBestFeatureToSplit(dataSet):
    # 每个样本的特征数量numFeatures
    numFeatures = len(dataSet[0]) - 1
    # 系统熵baseEntropy
    baseEntropy = calcShannonEnt(dataSet) 
    # 初始化信息增益bestInfoGain
    bestInfoGain = 0.0;
    # 初始化最好特征bestFeature
    bestFeature = -1  
    # 每个样本有2个特征，循环2次
    for i in range(numFeatures):
        # 取出每个样本的第i个位置上的特征值放进一个列表featList
        featList = [example[i] for example in dataSet] 
        # 防止featList重复uniqueVals
        uniqueVals = set(featList) 
        # 初始化新的熵 newEntropy
        newEntropy = 0.0
        # 遍历集合的值（2种：0和1）
        for value in uniqueVals:
            # 用每个特征value划分数据集
            subDataSet = splitDataSet(dataSet, i, value)
            # 把划分后的数据求概率
            prob = len(subDataSet) / float(len(dataSet))
            # 把划分后的数据计算系统熵，再求和newEntropy
            newEntropy += prob * calcShannonEnt(subDataSet)
        # 信息增益等于系统熵减去条件熵
        infoGain = baseEntropy - newEntropy 
        # 计算最大信息增益
        if (infoGain > bestInfoGain):  
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature


def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys(): classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


def createTree(dataSet, labels):
    # 取出所有类别放进列表classList
    classList = [example[-1] for example in dataSet]
    # 如果类别列表只有一种类别，直接返回这个类别
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    # 如果第一个样本长度为 1，
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel: {}}
    del (labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
    return myTree



myTree = createTree(dataSet, labels)

treePlotter.createPlot(myTree)
