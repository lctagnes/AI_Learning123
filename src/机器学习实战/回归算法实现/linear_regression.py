from numpy import *
import matplotlib.pyplot as plt

def loadDataSet(fileName):
    # 获取样本特征总数
    numFeat = len(open(fileName).readline().split('\t')) - 1
    dataMat = []
    labelMat = []
    fr = open(fileName)
    # 遍历每一行
    for line in fr.readlines():
        lineArr = []
        # 删除一行中以tab分割的前后空白符号
        curLine = line.strip().split('\t')
        # 遍历每一个特征，index：0，1
        for i in range(numFeat):
            # 每行的每个特征放到lineArr
            lineArr.append(float(curLine[i]))
        # 所有行的特征放到dataMat
        dataMat.append(lineArr)
        # 把目标变量放到labelMat
        labelMat.append(float(curLine[-1]))
    return dataMat, labelMat

def standRegres(xArr,yArr):
    # mat()函数将列表转换为矩阵，mat().T表示对矩阵转置
    xMat = mat(xArr)
    yMat = mat(yArr).T
    xTx = xMat.T * xMat
    # linalg.det(xTx)用来计算矩阵的行列式，如果为0，不可逆，不为0才可逆
    if linalg.det(xTx) == 0.0:
        print("This matrix is singular, cannot do inverse")
        return
    # 最小二乘法求得w的最优解
    ws = xTx.I * (xMat.T * yMat)
    return ws


dataMat, labelMat = loadDataSet("ex0.txt")
print(labelMat)
ws=standRegres(dataMat, labelMat)
xMat = mat(dataMat)
yMat = mat(labelMat)
x = xMat[:,1].flatten().A[0]
y = yMat.T[:,0].flatten().A[0]
plt.figure(figsize=(8,5))
plt.scatter(x, y, c='green')
plt.xlabel("X")
plt.ylabel("Y")
xCopy = xMat.copy()
xCopy.sort(0)
yHat = xCopy*ws
plt.plot(xCopy[:,1],yHat,c='red')
plt.show()
# ws = standRegres(dataMat,labelMat)
# xMat = mat(dataMat)
# yMat = mat(labelMat)
# fig = plt.figure(figsize=(8,5))
# ax = fig.add_subplot(111)
# ax.scatter(xMat[:,1].flatten().A[0],yMat.T[:,0].flatten().A[0])
# xCopy = xMat.copy()
# xCopy.sort(0)
# yHat = xCopy*ws
# ax.plot(xCopy[:,1],yHat)
# plt.show()






