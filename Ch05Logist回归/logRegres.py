'''
Created on Oct 27, 2010
Logistic Regression Working Module
@author: Peter
'''
from numpy import *

def loadDataSet():
    # 读取txt数据
    dataMat = []; labelMat = []
    fr = open('testSet.txt')
	# fr.readlines()读取所有的行
    for line in fr.readlines():
        lineArr = line.strip().split()    #
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])   # 保存数值
        labelMat.append(int(lineArr[2]))    # 保存标签
    return dataMat,labelMat

def sigmoid(inX):
    # inX: 因变量
	# sigmoid函数
    return 1.0/(1+exp(-inX))

def gradAscent(dataMatIn, classLabels):
    # 梯度上升算法，求得最大值，梯度下降求最小值
    dataMatrix = mat(dataMatIn)             #转化成矩阵convert to NumPy matrix
    labelMat = mat(classLabels).transpose()    #classLabels是行向量，现在转化成列向量convert to NumPy matrix
    m,n = shape(dataMatrix)    # 取得numpy的mat的纬度
    alpha = 0.001    # 步长
    maxCycles = 500   # 最大迭代次数
    weights = ones((n,1))    # 变量的初始权重
    for k in range(maxCycles):              #迭代500次求权重
        h = sigmoid(dataMatrix*weights)     #matrix mult
        error = (labelMat - h)              # 标签 与 sigmoid计算出来的标签的差获得logitic的错误率，然后，调整每个点的权重
        weights = weights + alpha * dataMatrix.transpose()* error # 梯度上升权重w的公式matrix mult
    return weights

def plotBestFit(weights):
    # 画图得到最合适的权重值
    import matplotlib.pyplot as plt
    dataMat,labelMat=loadDataSet()
    dataArr = array(dataMat)
    n = shape(dataArr)[0] 
    xcord1 = []; ycord1 = []
    xcord2 = []; ycord2 = []
    for i in range(n):
	    # 只有两类0 和 1
        if int(labelMat[i])== 1:
            xcord1.append(dataArr[i,1]); ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1]); ycord2.append(dataArr[i,2])
    fig = plt.figure()    # 一块画布对象
    ax = fig.add_subplot(111)    # 将画布分隔成1*1*1的画框区域
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')    # 在这个画框中画散点图
    ax.scatter(xcord2, ycord2, s=30, c='green') 
    x = arange(-3.0, 3.0, 0.1)   # 预测值，设定取三个变量x1 x2 x3 
    y = (-weights[0]-weights[1]*x)/weights[2] # 最佳的拟合直线，即分割线方程
    ax.plot(x, y)
    plt.xlabel('X1'); plt.ylabel('X2');
    plt.show()


# 随机上升梯度算法	
def stocGradAscent0(dataMatrix, classLabels):
    m,n = shape(dataMatrix)
    alpha = 0.01
    weights = ones(n)   #initialize to all ones
    for i in range(m):
        h = sigmoid(sum(dataMatrix[i]*weights))
        error = classLabels[i] - h
        weights = weights + alpha * error * dataMatrix[i]    # 关键的改变
    return weights


# 改进的随机上升梯度算法
def stocGradAscent1(dataMatrix, classLabels, numIter=150):
    m,n = shape(dataMatrix)
    weights = ones(n)   # 每个变量的初始化权重值
    for j in range(numIter):
        dataIndex = range(m)
        for i in range(m):
            alpha = 4/(1.0+j+i)+0.0001    #apha 随着每次迭代逐渐变小，越接近真实值的时候，步长越小 但是不会减小到0，因为有常数0.0001
            randIndex = int(random.uniform(0,len(dataIndex)))    # 随机选取样本来更新回归系数，能够减少周期性波动
            h = sigmoid(sum(dataMatrix[randIndex]*weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha * error * dataMatrix[randIndex]
            del(dataIndex[randIndex])    # 每次随机从列表中选取一个值，然后从该列表中删掉该值，再进行下一次迭代
    return weights

	
	
	
# ----------------------从疝气病预测病马的死亡率：对原始数据（有缺失）已经做了预处理
# 有缺失值时用0来替代，这样不影响变量系数的更新，适合logistic, sigmoid = 0.5 不具有倾向性不必考虑
# 因变量是分类变量，自变量是数值型变量，二者单位不同，用KNN分类不合适
def classifyVector(inX, weights):
    prob = sigmoid(sum(inX*weights))
    if prob > 0.5: return 1.0
    else: return 0.0

def colicTest():
    frTrain = open('horseColicTraining.txt'); frTest = open('horseColicTest.txt')
    trainingSet = []; trainingLabels = []
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')    # 每行的值以tab分裂
        lineArr =[]
        for i in range(21):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[21]))
    trainWeights = stocGradAscent1(array(trainingSet), trainingLabels, 1000)    # 随机梯度上升算法
    errorCount = 0; numTestVec = 0.0
    for line in frTest.readlines():
        numTestVec += 1.0    # 测试的行数
        currLine = line.strip().split('\t')
        lineArr =[]
        for i in range(21):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(array(lineArr), trainWeights))!= int(currLine[21]):
            errorCount += 1
    errorRate = (float(errorCount)/numTestVec)    # 错误率
    print "the error rate of this test is: %f" % errorRate
    return errorRate

def multiTest():
    numTests = 10; errorSum=0.0
    for k in range(numTests):
        errorSum += colicTest()
    print "after %d iterations the average error rate is: %f" % (numTests, errorSum/float(numTests))
        