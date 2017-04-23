#!/usr/bin/evn python
#-*-coding:utf-8-*-
'''
Created on Oct 19, 2010

@author: Peter
'''
from numpy import *

#斑点狗爱好者留言板；第1变量:实验样本；第2变量：类别标签集合

def loadDataSet():
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1]    #1 是侮辱性言论, 0 正常言论
    return postingList,classVec

'''
函数createVocabList(dataSet)会创建建一个包含在所有文档中出现的不重复词的列表，为
此使用了set数据类型，将词条列表输给set构造函数，set就会返回一个不重复词表
'''                
def createVocabList(dataSet):
	"""
	dataSet:原始数据，存放在二维数组中
	return：产生不重复的单词，存放在list中
	"""
    vocabSet = set([])  #用set创建一个空集列表
    for document in dataSet:
        vocabSet = vocabSet | set(document) #创建另个集合的并集(得到不重复的列表)，即第1个输入的列表和第0个输入的列表的并集
    return list(vocabSet)

'''
先考虑训练集
获得列表后，setOfWords2Vec()函数输入参数为词汇列表vocabList以及某个文档inputSet，
输出的是向量文档，向量的每一元素为 1或0 ，分别表示词汇表中的单词 在输人文档中是否出现 。
首先创 建一个和 词 汇 表等长的向 量， 并将其元素都设置为 0
遍历文档中的所有单词 ，如果出现了词汇表中的单词 ， 则将输出的文档向量中的 对应值设为 1

'''
def setOfWords2Vec(vocabList, inputSet):
	"""
	vocabList,是训练集中所有不重复的单词
	inputSet是每条留言组成的数组，那么word遍历每条留言
	return: 行为样本，列为单词，中间是样本对应的词频
	"""
    returnVec = [0]*len(vocabList)  
    for word in inputSet: 
        if word in vocabList: #该留言中，与词汇表中相对应的为1，不对应的为0 
            returnVec[vocabList.index(word)] = 1 #index取得vocabList中等于word的vocabList下标
        else: print "the word: %s is not in my Vocabulary!" % word
    return returnVec  #返回的是每条留言在词汇表中匹配的数量

'''
朴素贝叶斯分类训练函数
先看一看文章中的伪代码
输 人参数为文档矩阵trainMatrix，以及由 每篇文档类别标签所构成的向量trainCategory
首先 ，计算文档属 于侮辱性文档P(class=1)的 概率;二分类问题
遍历trainMatrix所有文档，一旦有侮辱性或正常词语，对应就会增加1
侮辱词汇个数为p1Num；正常为p0Denom

初始化值为1和2的原因
初始化，p0Num每条言论中负面言论都为1；正面言论都为1
利用 贝叶斯分类器对 文档进行分类时，要计算多个概率的 乘积以获得文档属 于某个类别的 概
如 果其中一个概率值为 0 , 那么最后的 乘积也为 0。为 降低这种 影响 ，
可以将所有词 的出现数初 始化为 1 ， 并将分母初 始化为 2

取对数的原因
由于大部分因子者3非常小用python尝 试相 乘许多很小的数，最后四 舍五 人后会得到 0
解决办法是对乘积取自然对数
通过求对数可以避免下 溢出或者浮点数舍入 导致的 错误。同时， 采用自然对 数进行处理不会有任何损失
'''

def trainNB0(trainMatrix,trainCategory):
	"""
	trainMatrix 文档矩阵， 训练集行是样本，列是单词，中间是词频
	trainCategory ：样本类别标签，1是侮辱性文档，0是正常文档
	return：每个单词在侮辱性文档和正常文档中的频率，以及侮辱性文档占所有文档的频率
	"""
    numTrainDocs = len(trainMatrix) #len()取数组trainMatrix的长度，即有多少条向量（取行数，即留言数）
    numWords = len(trainMatrix[0]) #trainMatrix[0] 取文档矩阵一行有多少个字，即词汇表有多少个字，trainMatrix是setOfWords2Vec的结果
    pAbusive = sum(trainCategory)/float(numTrainDocs)  #初始化，训练集中侮辱言论条数/总言论条数
    p0Num = ones(numWords); p1Num = ones(numWords)      
    p0Denom = 2.0; p1Denom = 2.0                   
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            #判断训练集为1是侮辱性文档，对应trainMatrix[i]为侮辱文档，那么该侮辱文档出现了词条，该词条加1
            p1Num += trainMatrix[i] 
            p1Denom += sum(trainMatrix[i]) #该文档词条总数再相加
        else:
            p0Num += trainMatrix[i] #正常词数加1
            p0Denom += sum(trainMatrix[i]) #该文档总词数加1
    p1Vect = log(p1Num/p1Denom)          #计算每个词在侮辱性文档中的概率
    p0Vect = log(p0Num/p0Denom)          #计算每个词在正常性文档中的概率
    return p0Vect,p1Vect,pAbusive

'''
4-3朴素贝叶斯函数

vec2Classify为
pClass1为训练集中侮辱性留言条的初始概率

'''
#贝叶斯决策理论，使用概率来分类
#vec2Classify新进入的样本，p0Vec第一类单词向量，p1Vec第二类单词向量，PClass1 第二类向量的概率
def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)    #element-wise mult
    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else: 
        return 0



def testingNB():
    #准备数据，并得到训练集词汇列表
    listOPosts,listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    trainMat=[]
    
    #计算一个文档中每个词汇出现为1，不出现为0
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))

    #计算每个词汇为侮辱词汇的概率和正常词汇的概率，pAb计算训练文档中侮辱文档的概率(作为先验概率)
    p0V,p1V,pAb = trainNB0(array(trainMat),array(listClasses))
    #要输入的留言条，检查它是否为侮辱留言
    testEntry = ['love', 'my', 'dalmation']
    #myVocabList这个训练词汇表 匹配上testEntry的为1 匹配不上的为0
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    #通过thisDoc中这三个词在侮辱文档中的概率之和 与在正常文档中的概率之和进行比较来判断
    print testEntry,'classified as: ',classifyNB(thisDoc,p0V,p1V,pAb)
    testEntry = ['stupid', 'garbage']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print testEntry,'classified as: ',classifyNB(thisDoc,p0V,p1V,pAb)

'''
reload(bayes)
bayes.testingNB()
'''


#4.5.4 准备数据：文档词袋模型，
#每个词在一个文档中不止出现一次，而在setOfWords2Vec()词集模型中，
#每个词只能出现一次
def bagOfWords2VecMN(vocabList, inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec

#########################4.6.2 使用朴素贝叶斯进行交叉验证######################

#4-5文本解析及完整的垃圾邮件测试函数
def textParse(bigString):    #输入一个大的字符串向量，输出字符串列表
    import re
    listOfTokens = re.split(r'\W*', bigString)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2] 

#交叉验证分类器 ， 对贝叶斯垃圾邮件分类器进行自动化处理   
def spamTest():
    docList=[]; classList = []; fullText =[] 
    for i in range(1,26):
        wordList = textParse(open('email/spam/%d.txt' % i).read()) #打开26个文档，并解析字符串
        docList.append(wordList) #每个文档作为一个列表。一共26个列表，组成一个数组，即将多个[], 添加为([[], [], []])
        fullText.extend(wordList) #将多个[]，扩展为([， ， ， ])
        classList.append(1) #每添加一个文档，就相当于增加一个类别
        wordList = textParse(open('email/ham/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)#创建词汇列表
    #创建测试集
    trainingSet = range(50); testSet=[]           
    for i in range(10):    #共有50封邮件，随机选择10封作为测试集，分类器中的概率计算只利用训练集中的的文档完成
        randIndex = int(random.uniform(0,len(trainingSet))) #uniform(low=0.0, high=1.0, size=1),随机均匀分布的产生size个数，默认产生1个
        testSet.append(trainingSet[randIndex]) #随机选择10个训练集，放入测试集中
        del(trainingSet[randIndex])  #删除训练集中被选择的10个测试集，这是为了交叉验证
     #
    trainMat=[]; trainClasses = []
    for docIndex in trainingSet:#对训练集文档遍历
        trainMat.append(bagOfWords2VecMN(vocabList, docList[docIndex])) #对每封邮件基于词汇表使用bagOfWords2VecMN函数构建词向量
        trainClasses.append(classList[docIndex])  #训练集的邮件类别
    p0V,p1V,pSpam = trainNB0(array(trainMat),array(trainClasses)) #计算训练集垃圾词汇的概率，垃圾邮件的概率
    errorCount = 0
    
    for docIndex in testSet:        #对测试邮件遍历
        wordVector = bagOfWords2VecMN(vocabList, docList[docIndex]) #
        if classifyNB(array(wordVector),p0V,p1V,pSpam) != classList[docIndex]: #测试集邮件被计算出来的类别与其真实的类别进行比较
            errorCount += 1 #邮寄错误分类加1，最后给出总的错误百分比
            print "classification error",docList[docIndex]
    print 'the error rate is: ',float(errorCount)/len(testSet)   #通过交叉验证，看一看这个分类器的准确率，这里的错误是将垃圾邮件误判为正常邮件
    #return vocabList,fullText

'''
bayes.spamTest()
bayes.spamTest()
'''

#################4.7使用朴素贝叶斯分类器从个人广告中获取区域倾向########################################


#4-6RSS源分类器及高频词去除函数
#calcMostFreq()遍历词汇表中的每个词并统计它在文本中出现的次数，然后根据出现次数从高到低对词典进行排序，最后返回
#排序最高的30个单词
def calcMostFreq(vocabList,fullText):
    import operator
    freqDict = {}
    for token in vocabList:
        freqDict[token]=fullText.count(token)
    sortedFreq = sorted(freqDict.iteritems(), key=operator.itemgetter(1), reverse=True) 
    return sortedFreq[:30]       

#localWords()使用两个RSS源作为参数，与spamTest()几乎相同，区别是一个是访问RSS源一个是访问文件，
def localWords(feed1,feed0):
    import feedparser
    docList=[]; classList = []; fullText =[]
    minLen = min(len(feed1['entries']),len(feed0['entries']))
    for i in range(minLen):
        wordList = textParse(feed1['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1) #NY is class 1
        wordList = textParse(feed0['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)#create vocabulary
    top30Words = calcMostFreq(vocabList,fullText)   #remove top 30 words
    
    for pairW in top30Words:
        if pairW[0] in vocabList: vocabList.remove(pairW[0])
    trainingSet = range(2*minLen); testSet=[]           #create test set
    
    for i in range(20):
        randIndex = int(random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])  
    
    trainMat=[]; trainClasses = []
    
    for docIndex in trainingSet:#train the classifier (get probs) trainNB0
        trainMat.append(bagOfWords2VecMN(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V,p1V,pSpam = trainNB0(array(trainMat),array(trainClasses))
    
    errorCount = 0
    
    for docIndex in testSet:        #classify the remaining items
        wordVector = bagOfWords2VecMN(vocabList, docList[docIndex])
        if classifyNB(array(wordVector),p0V,p1V,pSpam) != classList[docIndex]:
            errorCount += 1
    print 'the error rate is: ',float(errorCount)/len(testSet)
    return vocabList,p0V,p1V


#4.7.2分析数据：显示地域相关的用词
#先对pSF与pNF进行排序，然后按照顺序将词打印出来。
#4-7最具表征性的词汇显示函数
#getTopWords()使用两个RSS源作为输入。然后训练并测试朴素贝叶斯分类器，返回使用的概率值，然后创建两个列表用于
#元祖存储，与之前返回排名最高的x个单词不同，这个离可以返回大于某个阀值的所有词。这些元组会按照他们的条件概率
#进行排序。
def getTopWords(ny,sf):
    import operator
    vocabList,p0V,p1V=localWords(ny,sf)
    topNY=[]; topSF=[]
    for i in range(len(p0V)):
        if p0V[i] > -6.0 : topSF.append((vocabList[i],p0V[i]))
        if p1V[i] > -6.0 : topNY.append((vocabList[i],p1V[i]))
    sortedSF = sorted(topSF, key=lambda pair: pair[1], reverse=True)
    print "SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**"
    for item in sortedSF:
        print item[0]
    sortedNY = sorted(topNY, key=lambda pair: pair[1], reverse=True)
    print "NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**"
    for item in sortedNY:
        print item[0]


'''
reload(bayes)
import feedparser
ny = feedparser.parse('http://newyork.craigslist.org/stp/index.rss')
sf = feedparser.parse('http://sfbay.craigslist.org/stp/index.rss')
vocabList, pSF, pNY = bayes.localWords(ny, sf)
bayes.getTopWords(ny, sf)
'''