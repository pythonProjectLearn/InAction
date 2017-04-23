#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
第2章knn
"""
#修改程序字符；添加程序路径
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
sys.path.append('/media/zhoutao/软件盘/workspace/python_project/src/machinelearninginaction/Ch02')

#查看当前目录
import os
print os.getcwd()
#切换当前目录
os.chdir('/media/zhoutao/软件盘/workspace/python_project/src/machinelearninginaction/Ch02')

import kNN
#只有import kNN后才能利用reload()调用程序kNN.pyc
reload(kNN)  

#########2.2.1从文本中解析数据  #################
datingDataMat, datingLables = kNN.file2matrix('datingTestSet.txt')
datingLables[0:20] 

#########2.2.2分析数据，Matplotlib创建散点图#############################
import matplotlib
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111) #三个1
ax.scatter(datingDataMat[:, 1], datingDataMat[:, 2])
ax.scatter(datingDataMat[:, 1], datingDataMat[:, 2],
           ax.scatter(datingDataMat[:, 1], datingDataMat[:, 2]))
plt.show()
plt.close()

################2.2.3 准备数据：归一化################
reload(kNN)
#autoNorm()数据标准化
normMat, ranges, minVals = kNN.autoNorm(datingDataMat)
normMat[:20]
ranges[:20]
minVals[:20]

#每修改一次kNN.py都要重新载入kNN
reload(kNN)
kNN.datingClassTest()

import numpy as np


################2.2.4测试算法：作为完整程序验证分类器#####################
#
kNN.datingClassTest()

#############2.25 使用算法:构建完整可用系统########################
kNN.classifyPerson()


################2.3 手写识别系统###################
#2.3.1 准备数据，将图像转化为测试向量
#测试img2vector函数
testVector = kNN.img2vector('testDigits/0_13.txt')












