#!/usr/bin/evn python
#-*-coding: utf-8-*-
from math import log
import operator
import os
os.getcwd()
os.chdir('E:\\workspace\\python_project\\src\\machinelearning\\Ch03')

import trees
reload(trees)
#下载数据
myDat,labels = trees.createDataSet()
#计算香浓熵
trees.calcShannonEnt(myDat)

#增加类别
myDat[0][-1] = 'maybe'

#3.1.2划分数据集
trees.splitDataSet(myDat, 1,1)
tress.splitDataSet(myDat,0,0)



















