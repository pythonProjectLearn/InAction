#!/usr/bin/evn python
#-*-coding:utf-8-*-

from numpy import *
import os
os.getcwd()
os.chdir('E:\\workspace\\python_project\\src\\machinelearning\\Ch04')

import bayes
#listOpsts 是留言板数组词汇；listClasses分类0 1
listOposts, listClasses = bayes.loadDataSet()
#将留言板数组创建为不重复的留言板集合
myVocabList = bayes.createVocabList(listOposts)
#listOposts[0]为正常言论的第一个数组
bayes.setOfWords2Vec(myVocabList, listOposts[0])

#遍历数组listOposts中每个列表，myVocabList是postinDoc不重复的词汇列表
#trainMat是将postinDoc数组每一个向量长度扩展成myVocabList的长度
#myVocabList被postinDoc数组中的每一个向量遍历匹配，匹配上的myVocabList中的词
#设为1，不匹配的为0
trainMat = []
for postinDoc in listOposts:
    trainMat.append(bayes.setOfWords2Vec(myVocabList, postinDoc))

#通过P0V得到每个词在正常文档下的概率 和P1V侮辱性文档下的概率
#通过比较就可以得出 哪些词汇可以作为侮辱性文档的特征词，
#哪些词汇可以作为正常言论的特征词
P0V, P1V , PAb = bayes.trainNB0(trainMat, listClasses)
reload(bayes)
bayes.testingNB()

#4.6贝叶斯过滤垃圾邮件
#4.6.1准备数据：切分文本
mySent = 'This book is the best book on python or M.L. I have ever laid eyes on.'
#不能用mySent.split('')空格符分隔
mySent.split()
#使用正则表达式切分句子
import re
#构造切分规则
regEx = re.compile('\\W*')
#\W 匹配不是字母，数字，下划线的字符
#\w匹配字母，数字，下划线
#* 匹配0次到多次
listOfTokens = regEx.split(mySent)
#去掉空字符串，可以通过返回字符串大于0的字符串
[tok for tok in listOfTokens if len(tok)>0 ]
#.lower()改为小写；.upper()改为大写
[tok.lower() for tok in listOfTokens if len(tok)>0 ]
#看一封完整邮件的处理结果
emailText = open('./email/ham/6.txt').read()
listOfTokens = regEx.split(emailText)

#4.6.2测试算法:使用朴素贝叶斯进行交叉验证
import bayes
reload(bayes)

#4.7 使用朴素贝叶斯分类器从个人广告中获取区域倾向

























