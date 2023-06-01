# -*- coding: utf-8 -*-
"""
Created on Fri Jun 17 15:42:40 2016

@author: zang

QQ:1537171958
"""
from numpy import zeros  
from scipy.linalg import svd
import matplotlib.pyplot as plt
import string
from math import log
import numpy as np
from numpy import asarray, sum
import seaborn as sns
import pandas as pd


class LSA(object):
  
    def __init__(self, stopwords, ignorechars):
        """
        构造函数
        :param stopwords: 停用词
        :param ignorechars: 忽略
        """
        self.stopwords = stopwords
        self.ignorechars = ignorechars
        self.wdict = {}
        self.dcount = 0

    def parseDoc(self, doc):
        """
        解析文档
        :param doc:
        :return:
        """
        words = doc.split()
        for w in words:
            # 去除特殊字符

            w = w.lower().translate(str.maketrans('', '', self.ignorechars)).strip()
            if w == "":
                continue
            elif w in self.stopwords:
                continue
            elif w in self.wdict:
                self.wdict[w].append(self.dcount)
            else:
                self.wdict[w] = [self.dcount]
        self.dcount += 1
    
    def buildMwd(self):
        """
        构建Mwd
        :return:
        """
        self.keys = [k for k in self.wdict.keys() if len(self.wdict[k]) > 1]
        self.keys.sort()
        self.Mwd = zeros([len(self.keys), self.dcount])
        for i, k in enumerate(self.keys):
            for d in self.wdict[k]:
                self.Mwd[i, d] += 1
  
    def printMwd(self):
        print(self.Mwd)
    
    def TFIDF(self):
        """

        :return:
        """
        WordsPerDoc = sum(self.Mwd, axis=0)
        print(WordsPerDoc)
        DocsPerWord = sum(asarray(self.Mwd > 0, 'i'), axis=1) # i词出现在多少文档里
        print(DocsPerWord)
        rows, cols = self.Mwd.shape
        for i in range(rows):
          for j in range(cols):
            self.Mwd[i, j] = (self.Mwd[i, j] /WordsPerDoc[j]) * log(float(cols) / DocsPerWord[i])
  
    def calcSVD(self):
        """

        :return:
        """
        self.U, self.S, self.Vt = svd(self.Mwd)
        #print self.S
        targetDimension = 3
        self.U2 = self.U[0:, 0:targetDimension]
        self.S2 = np.diag(self.S[0:targetDimension])
        self.Vt2 = self.Vt[0:targetDimension, 0:]
        print("U:\n", self.U2)
        print("S:\n", self.S2)
        print("Vt:\n", self.Vt2)

    def plotSingularValuesBar(self):
        """

        :return:
        """
        y_value = (self.S*self.S)/sum(self.S*self.S)
        x_value = range(len(y_value))
        plt.bar(x_value, y_value, alpha=1, color='g', align="center")
        plt.autoscale()
        plt.xlabel("Singular Values")
        plt.ylabel("Importance")
        plt.title("The importance of Each Singular Value")
        plt.show()

    def plotSingularHeatmap(self):
        """

        :return:
        """
        labels = ["T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8", "T9"]
        rows = ["Dim1", "Dim2", "Dim3"]
        self.Vtdf_norm = pd.DataFrame(self.Vt2*(-1))
        #self.Vtdf_norm = (self.Vtdf - self.Vtdf.mean()) / (self.Vtdf.max() - self.Vtdf.min())
        self.Vtdf_norm.columns = labels
        self.Vtdf_norm.index = rows
        sns.set(font_scale=1.2)
        ax = sns.heatmap(self.Vtdf_norm, cmap=plt.cm.bwr, linewidths=.1, square=2)
        ax.xaxis.tick_top()
        plt.xlabel("Book Title")
        plt.ylabel("Dimensions")

if __name__ == "__main__":

    docs =\
    ["The Neatest Little Guide to Stock Market Investing",
    "Investing For Dummies, 4th Edition",
    "The Little Book of Common Sense Investing: The Only Way to Guarantee Your Fair Share of Stock Market Returns",
    "The Little Book of Value Investing",
    "Value Investing: From Graham to Buffett and Beyond",
    "Rich Dad's Guide to Investing: What the Rich Invest in, That the Poor and the Middle Class Do Not!",
    "Investing in Real Estate, 5th Edition",
    "Stock Investing For Dummies",
    "Rich Dad's Advisors: The ABC's of Real Estate Investing: The Secrets of Finding Hidden Profits Most Investors Miss"]
    
    stopwords = ['and', 'edition', 'for', 'in', 'little', 'of', 'the', 'to']
    ignorechars = string.punctuation
    lsaDemo = LSA(stopwords, ignorechars)
    for d in docs:
        lsaDemo.parseDoc(d)
    lsaDemo.buildMwd()
    lsaDemo.printMwd()
    #lsaDemo.TFIDF()
    lsaDemo.printMwd()
    lsaDemo.calcSVD()
    lsaDemo.plotSingularValuesBar()
    lsaDemo.plotSingularHeatmap()