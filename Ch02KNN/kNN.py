#!/usr/bin/python
# -*- coding: UTF-8 -*-
'''
Created on Sep 16, 2010
kNN: k Nearest Neighbors

Input:      inX: vector to compare to existing dataset (1xN)
            dataSet: size m data set of known vectors (NxM)
            labels: data set labels (1xM vector)
            k: number of neighbors to use for comparison (should be an odd number)
            
Output:     the most popular class label

@author: pbharrin
'''
# 这样第输入方式，导致整个程序都不用numpy作为函数前缀（例如zeros）
from numpy import *
import operator
from os import listdir


# 选出k个测试样本，
# 首先得到第i个测试样本的类别voteIlabel
# 构建类别，以及k各样本该类别的数量 的字典
# 按照类别的数量，从大到小（倒序）排列字典
# 取得类别数量最多的那个类
def classify0(inX, dataSet, labels, k):
    '''

    :param inX: （int）测试集
    :param dataSet: （numpy.array）训练集
    :param labels: 数据集dataSet所在的类别
    :param k: k表示邻居个数
    :return:
    '''
    # shape设置dataSet的属性（维度行数， 每行代表一个样本，每行对应一个类别）
    dataSetSize = dataSet.shape[0]   # 样本量
    # 计算欧式距离
    # tile（numpy中函数）重复inx，tile(a, (2,3))表示输出一个2行3列的数组，a在每一行重复了3次，一共两行
    # tile是行向量（一个测试样本）重复训练样本的个数，相减，求得欧式距离
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet  #
    sqDiffMat = diffMat**2  # 平方
    # argsort(){numpy中的函数}排序（倒序），距离从大到小
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5 
    
    sortedDistIndicies = distances.argsort()    # 从大到小距离排序
    classCount={}          #存储类别
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

def createDataSet():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group, labels


###################################################################
# 2.2使用k近邻算法改进约会网站的配对效果
# 2.2.1 准备数据从文本文件中解析数据
'''
样本中包含3个特征：1 飞行常客里程数 2 玩游戏所耗时间百分比 3 每周消费冰淇淋公升数
1 将文本记录到numpy的解析程序;
2 classLabelVector.append(int(listFromLine[-1]))
里面的int去掉，明显不能这样转换，因为都是文字 ;
3 zeros是numpy中的函数，由于采用from numpy import *输入方式，
所以zeros不用前缀numpy

'''
# file2matrix函数 ：处理输入格式问题
# 输入为文件名 字符串， 输出为 训练样本矩阵和 类标签向量。
# fr.readlines() 按行读取， 每行代表一个训练样本，读取所有的行，
# 按行读取会把txt中的空格符和末尾的换行符读出来，因此要用split()  split("\t")
# 按行放置数据，将读取的数据放入returnMat列表中

#读取文件数据
def file2matrix(filename):
    '''
    :param filename: (txt文件)datingTestSet.txt
    :return:
    '''
    fr = open(filename)  # 打开文件
    numberOfLines = len(fr.readlines())         # 读取所有的行，得到行数
    # 以0填充的特征矩阵， zeros((10,3)) 得到10行3列的数组（矩阵）
    returnMat = zeros((numberOfLines, 3))
    classLabelVector = []                       # 存储矩阵的列名称

    index = 0
    for line in fr.readlines():
        line = line.strip()  # 去掉两端的空格
        listFromLine = line.split('\t')    # 用tab(\t)键得到line的整行数据分割成一个元素列表。
        returnMat[index, :] = listFromLine[0:3]  # 前三列数自变量，最后一列
        classLabelVector.append(listFromLine[-1])  # 将最后一列（类别名）添加到向量中
        index += 1
    return returnMat,classLabelVector #返回一个元组，这个元组中包含有列表returnMat
'''
reload(kNN)
datingDataMat, datingLabels = kNN.file2matrix('datingTestSet.txt')

2.2.2 分析数据，使用Matplotlib创建散点图
import matplotlib
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111) #三个1
ax.scatter(datingDataMat[:,1], datingDataMat[:,2])
plt.show()

'''



'''
2.2.3 归一化特征值，使数据在0-1之间
#在Numpy库中linalg.solve(matA, matB)矩阵除法
'''    
def autoNorm(dataSet):
    minVals = dataSet.min(0)    #取每列的最小值第0位
    maxVals = dataSet.max(0)    #取每列的最大值第0位
    ranges = maxVals - minVals  #得到区间
    normDataSet = zeros(shape(dataSet)) #shape取得数据的维度dim，并设置矩阵为0
    m = dataSet.shape[0] #shape[0]取行数，shape[1]取列数
    normDataSet = dataSet - tile(minVals, (m,1)) #减去最小值
    normDataSet = normDataSet/tile(ranges, (m,1))   #除以区间大小
    return normDataSet, ranges, minVals


'''
2-4分类器针对约会网站的测试代码
'''   
def datingClassTest():
    hoRatio = 0.1      #抽取前 10%
    datingDataMat,datingLabels = file2matrix('datingTestSet2.txt')       #把文件中的数据读出来
    normMat, ranges, minVals = autoNorm(datingDataMat) #对数据标准化，都是list类型数据
    m = normMat.shape[0]  #行数
    numTestVecs = int(m*hoRatio) # 取前10%的样本
    errorCount = 0.0 #初始化错误计数
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[1,:],normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],3) #返回的是字符，所以不能用%d
        print "the classifier came back with: %s, the real answer is: %s" % (classifierResult, datingLabels[i])
        if (classifierResult != datingLabels[i]): errorCount += 1.0
    print "the total error rate is: %f" % (errorCount/float(numTestVecs))
    print errorCount

#在python中不能用转化float
def classifyPerson():
    resultList = ['not at all', 'in small doses','in large doses']
    percentTats = float(raw_input(\
    "percentage of time spent playing video games?"))
    ffMile = float(raw_input("frequent fliter miles earned per year?"))
    iceCream = float(raw_input("liters of ice cream consumed per year?"))
    datingDataMat, datingLables = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat) #标准化
    inArr = array([ffMile, percentTats, iceCream]) # 
    classifierResult = classify0((inArr-minVals)/ranges, normMat, datingLables,3) #分类
    print "you will probably like this person:", \
    resultList[int(classifierResult) - 1] #int()使字符变成数字减1 ，代表三类

###########################################################
#2.3.1将图像转化为测试向量  
#将图像32*32的二进制图像转化为1*1024的行向量，存储在returnVect里，所以要初始化为0
#所以存储的对象都要初始化为0
#每个filename 是一个手写字的文件，  
def img2vector(filename):
    returnVect = zeros((1,1024)) 
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline() #每次读取一行
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j]) #每次读一个元素j进取
    return returnVect

############################################################
#2.3.2测试算法：使用K近邻算法识别手写数字
def handwritingClassTest():
    hwLabels = [] #存储训练集的类别，首先要初始化
    
   #针对训练集    
    trainingFileList = listdir('trainingDigits') #获取训练集文件目录的名称返回list列向量，os.listdir()
    m = len(trainingFileList) #文件的个数
    trainingMat = zeros((m,1024)) #组成矩阵（mat在numpy中表示多维数组或者矩阵），存储所有训练集二维数据
    #i表示每个训练样本文件      
    for i in range(m):
        fileNameStr = trainingFileList[i] #循环取出第i个训练样本
        fileStr = fileNameStr.split('.')[0]     #去掉文件名中的后缀 .txt，取文件名
        classNumStr = int(fileStr.split('_')[0])#文件名“_”前面的数字代表训练集的类别
        hwLabels.append(classNumStr) #将每个训练样本类别添加到训练样本类别中
        #构建每个训练样本的路径 ，将训练集转化成向量，每个向量对应它自己的类别       
        trainingMat[i,:] = img2vector('trainingDigits/%s' % fileNameStr) 
    
    #针对测试集 ， 同训练集的处理方式一样  
    testFileList = listdir('testDigits')        #iterate through the test set
    errorCount = 0.0 #存放分类错误的实例个数
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]     #take off .txt
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)
        
        #将训练集和测试集放入kNN.classify0()函数中,进行分类
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print "the classifier came back with: %d, the real answer is: %d" % (classifierResult, classNumStr)
        if (classifierResult != classNumStr): errorCount += 1.0
    
    print "\nthe total number of errors is: %d" % errorCount
    print "\nthe total error rate is: %f" % (errorCount/float(mTest))
