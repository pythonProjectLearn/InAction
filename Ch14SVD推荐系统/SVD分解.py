#coding:utf-8
#利用python实现svd
#linalg是numpy的线性代数工具
from numpy import *
U, Sigma,VT = linalg.svd([[1,1], [7,7]])
#Sigma返回的是array格式，其实应该是一个矩阵， 但是它实际上是矩阵的对角线元素

#还原原始矩阵的近似矩阵，对角线元素太小的直接忽视，以较小的维数保留较多的信息
#构建三个元素的矩阵
Sig3 = mat([[Sigma[0], 0, 0], [Sigma[1], 0, 0], [Sigma[0], 0, 0]])
U[::3]*Sig3*VT[:3,:]

#