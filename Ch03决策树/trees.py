#!/usr/bin/evn python
#-*-coding: utf-8-*-
'''
Created on Oct 12, 2010
Decision Tree Source Code for Machine Learning in Action Ch. 3
@author: Peter Harrington
'''

#首先学习如何使用python计算熵

from math import log #从math包中引出 log()函数
import operator

#鱼鉴定数据集， 第一列代表surfacing（鱼鳞）， 第二列代表flippers（鱼鳍），第三列判断这个动物是不是鱼
def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing','flippers']
    #change to discrete values
    return dataSet, labels

##################3-1计算给定数据集的香浓熵#####################################
#计算熵，选择根节点的特征
#dataSet 表示样本集和
def calcShannonEnt(dataSet):
	#数据样本数量
    numEntries = len(dataSet)	
	#创建一个空字典，用来装每个样本的类别（key）和特征向量的（value）
    labelCounts = {} 

	#遍历每个样本，每次取一行数据，featVec变成了一维的list
    for featVec in dataSet:
		#得到一个样本的标签(类别)
        currentLabel = featVec[-1] 
        # 如果当前键值(类别)不存在，则扩展字典并将当前键加入字典，并初始化为0        
        if currentLabel not in labelCounts.keys(): 
            labelCounts[currentLabel] = 0
		
		#每个键值都记录了当前类别出现的次数(不重复的为1，重复的增加1)
        labelCounts[currentLabel] += 1
	
    # 香农熵初始化	
    shannonEnt = 0.0
	
    #使用所有类标签发生的频率计算类别出现的概率  	  
    for key in labelCounts:
		# 一个类出现的次数除以总的样本数，得到每个类别的概率
        prob = float(labelCounts[key])/numEntries 
        shannonEnt -= prob * log(prob,2) #log base 2 计算每个类别的熵
    return shannonEnt #返回熵

############################3.1.2划分数据集########################################
#######3-2 按照特征划分数据集,计算信息增益
#对每个特征划分数据集的结果计算一次信息熵， 然后比较信息熵哪个最大，来判断按照哪个特征划分数据集是最好的划分方式
#dataSet待划分数据集, axis每个样本第i列的特征（自变量）, value需要返回的特征值（计算特征为value的信息熵）
def splitDataSet(dataSet, axis, value):
    """
	dataSet:数据集
	axis：第几列特征，即自变量
	value：自变量的特征值
	"""
	# 存放按照某个特征划分数据，分成左侧和右侧的数据	
    retDataSet = [] 
    
	# 遍历每个样本
    for featVec in dataSet: 
		# 第axis列的特征与我们要求的value特征相等的样本挑出来
        if featVec[axis] == value: 
			# 取出前axis列的数据
            reducedFeatVec = featVec[:axis]
			# 取axis列后的数据
            reducedFeatVec.extend(featVec[axis+1:])			
			# 
            retDataSet.append(reducedFeatVec)
    return retDataSet


# 3-3选择最好的数据集划分方式
def chooseBestFeatureToSplit(dataSet):
	"""
	return:划分数据集的最好的特征变量的位置i（不是特征变量的标签）
	"""
	# 获得特征变量的个数，因为最后一列保存样本的数据标签（类别）所以减1
    numFeatures = len(dataSet[0]) - 1
	
	# 计算整个数据集的熵（总的熵）	
    baseEntropy = calcShannonEnt(dataSet)
	
	# 信息增益， 和最好的特征
    bestInfoGain = 0.0; bestFeature = -1
	
    # 遍历每个特征
	for i in range(numFeatures):
	
		# 取得第i个特征变量的，所有样本的的特征值，即dataSet的第i列的所有数据
        featList = [example[i] for example in dataSet]
		
        # 因为要按照特征值的进行划分二叉树，所有不能有重复值
        uniqueVals = set(featList)  
        
        # 按照第i个特征划分时获得的香农熵，存放在newEntropy中		
        newEntropy = 0.0
		
        
		# 对第i个特征变量的某个值，划分二叉树
		# 计算每种划分方式的信息熵
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet)/float(len(dataSet))		
            newEntropy += prob * calcShannonEnt(subDataSet)
		
		# 计算信息增益
		# 整个数据集的信息熵，减去以第i个特征变量划分获得的香农熵
		# 就得到信息增益		
        infoGain = baseEntropy - newEntropy
		
		#比较信息增益与当前最好的信息增益对比
        if (infoGain > bestInfoGain):       
            #信息增益比越大越，那么按照这个特征分类越好
			bestInfoGain = infoGain			
            bestFeature = i
			
    # 返回最好的特征
	return bestFeature                      

'''
import os
os.chdir('/media/zhoutao/软件盘/workspace/python_project/src/machinelearninginaction')
reload(trees)
mydata, labels = trees.createDataSet()
trees.chooseBestFeatureToSplit(mydata)
mydata
'''

#####################3.1.3递归构建决策数##############################
'''
classList = [example[-1] for example in mydata]
'''
def majorityCnt(classList):
	"""
	classList：样本的类别标签（数据有重复）
	return：频数最高的类别
	"""
	# 存放所有样本的类别，以及类别的频数
    classCount={}
	
    #计算
    for vote in classList:
        if vote not in classCount.keys(): 
            classCount[vote] = 0
        classCount[vote] += 1
		
    #利用operator操作键值排序字典，按照value排序，iteritems()得到vaule， 返回list
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
	
	#得到类标签频数最高的类标签
    return sortedClassCount[0][0]  
	

#######3-4创建树的函数代码
#三个return是等价的
def createTree(dataSet, labels):
	"""
	dataSet：原始数据集
	labels: 特征变量的名称
	"""
	#存放样本的所有样本的类别（类别有重复）， 是鱼 /非鱼两类
    classList = [example[-1] for example in dataSet]
    
    # ------------编写递归，首先要给出递归的终止条件
	# 递归终止条件，即最极端的两种递归情况
	
    # 第一个终止条件
	# 假设所有类标签完全相同，所以classList[0]就能取得这个数据集的唯一的类别标签
	# classList.count()就能获得这个唯一类别的个数，
	# 如果它们相等，那么就确认这个数据集确实只有一个类别，那么就返回这个类别
    if classList.count(classList[0]) == len(classList): 
        return classList[0]
		
    #第二个终止条件，使用完了所有特征，仍然不能将数据集划分成仅包含一类别的分组，
	#由于不能返回唯一类标签，所以这里返回类标签最多的类别作为返回值
    if len(dataSet[0]) == 1: #
        return majorityCnt(classList)
    # -------------------------------------
	
	 
    #通过计算信息增益得到最好的划分特征变量
	#bestFeat是特征变量的位置，不是特征变量的标签
    bestFeat = chooseBestFeatureToSplit(dataSet)
	
	# 通过labels得到这个特征变量的标签
    bestFeatLabel = labels[bestFeat]
	
	# 存放树，最好的特征变量标签作为key，特征变量中被划分的点作为子字典中的key
	# 类别作为value
    myTree = {bestFeatLabel:{}}
	
	# 首先从标签集合中删除这个最好的特征变量
    del(labels[bestFeat]) 
	
	# 取出这个最好特征变量的值（就是这个变量的列）
    featValues = [example[bestFeat] for example in dataSet] 
	
	# 因为要对这个最好变量进行左右侧划分，所以要得到不重复的值
    uniqueVals = set(featValues) 
    
	# 对该变量的每个值遍历
    for value in uniqueVals:
		# 删除了被选中的最好的特征变量之后，剩下的特征变量    
        subLabels = labels[:]		
		# 递归调用createTree(),对每个特征变量进行划分
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value),subLabels)
    return myTree                            

'''
reload(trees)
mydata, labels = trees.createDataSet()
myTree = trees.createTree(mydata, labels)
myTree 
'''

############3.3.1测试算法：使用决策树执行分类#####################
#3-8使用决策树的分类函数
#    
def classify(inputTree,featLabels,testVec):
	"""
	inputTree: mytree
	featLabels: 特征变量标签
	testVec： 要分类的新样本
	"""
    firstStr = inputTree.keys()[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr) #使用index方法查找当前列表中第一个匹配firstStr变量的元素。
    key = testVec[featIndex]  
    valueOfFeat = secondDict[key]
	
    #递归遍历整棵树，比较testVec变量中的值与树节点的值，如果达到叶子节点，则返回当前节点的分类标签。
    if isinstance(valueOfFeat, dict): 
        classLabel = classify(valueOfFeat, featLabels, testVec)
    else: 
		classLabel = valueOfFeat
	
    return classLabel

'''
myData, labels = trees.createDataSet()
#第一节名为 no surfacing ,它有两个节点：一个名字为0 的叶节点，类标签为no
#另一个为判断节点flippers,此处进入递归调用，flippers有两个子节点
myTree = treePlotter.retrieveTree(0)
myTree
trees.classify(myTree, labels, [1,0])
trees.classify(myTree, labels, [1,1])
'''

######决策树的存储
#在每次执行分类时调用已经构造好的决策树。pickle模块用来序列化对象，可以在磁盘上保存对象，并在需要的时候读出来。
#任何对象都可以序列化，包括字典在内
def storeTree(inputTree,filename):
    import pickle
    fw = open(filename,'w')
    pickle.dump(inputTree,fw)
    fw.close()
    
def grabTree(filename):
    import pickle
    fr = open(filename)
    return pickle.load(fr)
    
'''
trees.storeTree(myTree, 'classifierStorage.txt') #存储到classifierStorage.txt中
trees.grabTree('classifierStorage.txt') #读出来
'''

########################使用决策树预测隐形眼镜类型#####################
fr = open('lenses.txt')
lenses = [inst.strip().split('\t') for inst in  fr.readlines()]
lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate'] #特征，列名
lensesTree = trees.createTree(lenses, lensesLabels) #创建决策树，存放在字典中
treePlotter.createPlot(lensesTree) #画决策树