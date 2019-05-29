

import numpy as np
import operator


# 创建数据集
def createDataSet():
    matrix = np.array([[1.0, 1.1], [1.0, 1.0], [0.0, 0.0], [0.0, 0.1]])
    classVector = ['A', 'A', 'B', 'B']
    return matrix, classVector


matrix, classVector = createDataSet()


# 算法封装
def classify(inX, matrix, classVector, k):
    dataSetSize = matrix.shape[0]
    diffMat = np.tile(inX, (dataSetSize, 1)) - matrix
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)
    distance = sqDistances ** 0.5
    sortDistIndicies = distance.argsort()
    classCount = {}
    for i in range(k):
        voteLabel = classVector[sortDistIndicies[i]]
        classCount[voteLabel] = classCount.get(voteLabel, 0) + 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    print(sortedClassCount[0][0])


classify([1, 1], matrix, classVector, 3)
