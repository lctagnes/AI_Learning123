# encoding: utf-8

from numpy import *


def loadDataSet():
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0, 1, 0, 1, 0, 1]  # 0 不是脏话 1 是脏话
    return postingList, classVec


dataSet, classVec = loadDataSet()


def createVocabList(dataSet):
    # 定义一个集合vocabSet用来存储整个文档的所有单词，无重复
    vocabSet = set([])
    # 遍历整个数据集的每个样本document
    for document in dataSet:
        # 集合的并集
        vocabSet = vocabSet | set(document)  
    # 返回一个有序列表
    return list(vocabSet)


# print createVocabList(dataSet)


def setOfWords2Vec(vocabList, inputSet):
    # 初始化词汇列表returnVec：32个0
    returnVec = [0] * len(vocabList) 
    # 每个样本的每个单词
    for word in inputSet:
        # 如果该单词出现在词汇列表里
        if word in vocabList:
            # 把该单词的所在位置标记为 1
            returnVec[vocabList.index(word)] = 1
    return returnVec

# trainMatrix 词向量 集合
def trainNB0(trainMatrix, trainCategory):  
    # 样本（文件）个数numTrainDocs
    numTrainDocs = len(trainMatrix)  # 6
    # 无重复单词数numWords
    numWords = len(trainMatrix[0])  # 32
    # 脏话样本的概率pAbusive，因为非脏话都是0，所以sum是脏话的概率
    pAbusive = sum(trainCategory) / float(numTrainDocs)  
    # 单词出现次数的列表：初始化非脏话p0Num和脏话p1Num都是32个1
    p0Num = ones(numWords)  
    p1Num = ones(numWords)
    # 整个数据集单词出现总数，分母初始值p0Denom，p1Denom
    p0Denom = 2.0  
    p1Denom = 2.0
    # 把为1（脏话） 为0（非脏话） 的值全部计算，6个类别
    for i in range(numTrainDocs):
        # 如果是脏话
        if trainCategory[i] == 1:
            # 对脏话的词向量进行加和，一一对应相加放到p1Num
            p1Num += trainMatrix[i]
            # 脏话样本的单词总数p1Denom
            p1Denom += sum(trainMatrix[i])
        else:
            # 对非脏话的词向量进行加和，一一对应相加放到p0Num
            p0Num += trainMatrix[i]
            # 非脏话样本的单词总数p0Denom
            p0Denom += sum(trainMatrix[i])
    # 整个文档中有脏话p1Vect和无脏话p0Vect出现的概率
    p1Vect = log(p1Num / p1Denom)  
    p0Vect = log(p0Num / p0Denom)
    return p0Vect, p1Vect, pAbusive


trainMatrix = []
# 我的词汇集
myVocabList = createVocabList(dataSet)
for postinDoc in dataSet:
    trainMatrix.append(setOfWords2Vec(myVocabList, postinDoc))
# print trainMatrix

p0Vect, p1Vect, pAbusive = trainNB0(trainMatrix, classVec)

# vec2Classify待测数据的词向量 
# 脏话类别的列表p1Vec和无脏话类别的列表p0Vec
# pClass1训练集脏话的概率
# 使用算法：
#       将乘法转换为加法
#       乘法：P(C|F1F2...Fn) = P(F1F2...Fn|C)P(C)/P(F1F2...Fn)
#       加法：P(F1F2...Fn|C)*P(C) = P(F1|C)*P(F2|C)....P(Fn|C)*P(C) -> log(P(F1|C))+log(P(F2|C))+....+log(P(Fn|C))+log(P(C))
def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    # 脏话的概率p1 贝叶斯的分子
    # vec2Classify * p1Vec是两个array，代表对应元素相乘，意思是把每个词和对应的概率关联起来
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)
    # 非脏话的概率p0 贝叶斯的分子
    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0


def testingNB():
    listOPosts, listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    trainMat = []
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    p0V, p1V, pAb = trainNB0(array(trainMat), array(listClasses))
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry, '类别为: ', classifyNB(thisDoc, p0V, p1V, pAb))
    testEntry = ['stupid', 'garbage']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry, '类别为: ', classifyNB(thisDoc, p0V, p1V, pAb))

testingNB()