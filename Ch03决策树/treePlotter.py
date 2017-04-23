#!/usr/bin/evn python
#-*-coding: utf-8-*-
'''
Created on Oct 14, 2010

@author: Peter Harrington
'''
import matplotlib.pyplot as plt

#定义文本框decisionNode leafNode和箭头arrow_args, 都是字典格式
#定义注释方框；父节点的方框是decisionNode；子节点的方框是leafNode
#boxstyle表示方框的类型，fc表示方框的大小
decisionNode = dict(boxstyle="sawtooth", fc="0.8") 
leafNode = dict(boxstyle="round4", fc="0.8")
#定义一个剪头的类型
arrow_args = dict(arrowstyle="<-")


######################构造注解树#####################
#3-6获取叶节点的数目和树的层数
#遍历整棵树，累计叶子节点的个数，并返回值
def getNumLeafs(myTree):
	# 初始化节点的数目
    numLeafs = 0 	
	# 父节点
    firstStr = myTree.keys()[0] 	
	#子节点
    secondDict = myTree[firstStr]  
    #如果子节点是字典类型，则该节点也是一个判断节点，需要递归调用getNumLeafs()函数
    for key in secondDict.keys():
		#type函数判断子节点是否为字典类型
        if type(secondDict[key]).__name__=='dict':
            #secondDict[key]表示子树
			numLeafs += getNumLeafs(secondDict[key]) 
        #最后没有字典了，说明到了最底层，此时叶节点要加1
		else:   
			numLeafs +=1 
    return numLeafs


def getTreeDepth(myTree):
    """
	计算树的深度。
	计算遍历过程中遇到判断节点的个数.该函数的终止条件是叶子节点，
    一旦到达叶子节点，则从递归调用中返回，并将计算树深度的变量加1
	myTree: 树是由dict构造
	"""
    maxDepth = 0
    firstStr = myTree.keys()[0] #父（根）节点
    secondDict = myTree[firstStr]  #所有子节点放在secondDict中
	
	# 遍历第一层的所有的子节点（节点是放在字典的key中）
    for key in secondDict.keys():
	    # type函数判断子节点是否为字典类型，就是为了判断是否到了叶节点
        if type(secondDict[key]).__name__=='dict':
		    # 递归一次，增加1个深度
            thisDepth = 1 + getTreeDepth(secondDict[key]) 
        # 如果到了叶节点，就是一个数值，不是字典类型，那么该层深度为1
		else:   
		    thisDepth = 1
        # maxDepth是用来存储树的深度，thisDepth是存储每次递归时的树的深度
        # thisDepth相对于中间变量	
        if thisDepth > maxDepth: 
		    maxDepth = thisDepth
    return maxDepth


#绘制带箭头的注解

def plotNode(nodeTxt, centerPt, parentPt, nodeType):
	"""
	nodeTxt箭头处的文本；
	centrePt箭头的坐标（在平面上（x=0.5,y=0.1））；
	parentPt箭尾的坐标(x = 0.1， 0.5)
	nodeType = decisionNode将文本nodeTxt添加到这个方框里
	xy是尾部坐标，xytext箭头处的坐标；arrowprops箭头；
	va ha 行和竖的位置
	bbox 方框的类型
	"""
    createPlot.ax1.annotate(nodeTxt, xy=parentPt, xytext=centerPt, 
                            xycoords='axes fraction',
                            textcoords='axes fraction',
                            va="center", ha="center", 
                            bbox=nodeType, 
                            arrowprops=arrow_args )

'''
#首先创建一个新图形并清空绘图区，然后在绘图区上绘制两个代表不同类型的树节点，
def createPlot1():
    fig = plt.figure(1, facecolor='white')
    fig.clf() #
    createPlot1.ax1 = plt.subplot(111, frameon=False) #ticks for demo puropses 
    #绘制第一个箭头和注释
    plotNode('a decision node', (0.5, 0.1), (0.1, 0.5), decisionNode) #
    #绘制第二个箭头带注释
    plotNode('a leaf node', (0.8, 0.1), (0.3, 0.8), leafNode)
    plt.show()
'''
###########plotTree函数 

#在父子节点间填充文本信息
 
def plotMidText(cntrPt, parentPt, txtString):
	"""
	#parenetPt 每一层的父节点，
	cntrPt 每一层的子节点，
	txtString 文本信息  
	"""
    xMid = (parentPt[0]-cntrPt[0])/2.0 + cntrPt[0]
    yMid = (parentPt[1]-cntrPt[1])/2.0 + cntrPt[1]
    #createPlot.ax1对象是plt.subplot(111, frameon=False, **axprops)这个画布，
	#在画布上画方框中的文本
    createPlot.ax1.text(xMid, yMid, txtString, va="center", ha="center", rotation=30)


def plotTree(myTree, parentPt, nodeTxt):
    """
	#parentPt父节点
	nodeTxt子节点
	"""
    numLeafs = getNumLeafs(myTree)  #首先计算树的宽度
    depth = getTreeDepth(myTree)    #计算树的深度
    firstStr = myTree.keys()[0]     #根节点的文本放入其中
    #plotTree.xOff全局变量，追踪已经绘制的节点位置，
	# plotTree.yOff放置下一个节点的恰当位置；
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs))/2.0/plotTree.totalW, plotTree.yOff)
    plotMidText(cntrPt, parentPt, nodeTxt)  #画方框中的文本信息
    plotNode(firstStr, cntrPt, parentPt, decisionNode) #画箭头，并且在箭头上带有注解
    secondDict = myTree[firstStr]
	#因为是自顶向下绘制图形，所以要依次递减y坐标
    plotTree.yOff = plotTree.yOff - 1.0/plotTree.totalD  
	
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':
            plotTree(secondDict[key],cntrPt,str(key))
        else:   #it's a leaf node print the leaf node
            plotTree.xOff = plotTree.xOff + 1.0/plotTree.totalW
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
    plotTree.yOff = plotTree.yOff + 1.0/plotTree.totalD

#if you do get a dictonary you know it's a tree, and the first element will be another dict


#createPlot()是主函数，它调用了plotTree()和plotTree又依次调用了前面的函数和plotMidText()
def createPlot(inTree):
	#画一副图，背景色为白色
    fig = plt.figure(1, facecolor='white') 
    fig.clf()
	#初始化左支 和 右支
    axprops = dict(xticks=[], yticks=[]) 
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)
    #createPlot.ax1 = plt.subplot(111, frameon=False) #ticks for demo puropses 
    plotTree.totalW = float(getNumLeafs(inTree))  #树的宽度
    plotTree.totalD = float(getTreeDepth(inTree)) #树的深度
    plotTree.xOff = -0.5/plotTree.totalW #x的有效范围是0~1
    plotTree.yOff = 1.0  #y的范围是0~1
    plotTree(inTree, (0.5,1.0), '') #
    plt.show()

'''
reload(treePlotter)
myTree = treePlotter.retrieveTree(0) #第0颗树
treePlotter.createPlot(myTree)
'''
#输出预先存储的树信息，避免了每次测试代码时都要从数据中创建数的麻烦，主要用于测试。
def retrieveTree(i):
    listOfTrees =[{'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}},
                  {'no surfacing': {0: 'no', 1: {'flippers': {0: {'head': {0: 'no', 1: 'yes'}}, 1: 'no'}}}}
                  ]
    return listOfTrees[i]

'''
reload(treePlotter)
treePlotter.retrieveTree(1)
myTree = treePlotter.retrieveTree(0)#得到树0
treePlotter.getNumLeafs(myTree) #等于树0的叶子节点数
'''
