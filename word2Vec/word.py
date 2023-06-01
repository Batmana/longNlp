# -*- coding: utf-8 -*-
"""
Word2Vec模型
"""
import os
import logging
import string
import re
import jieba
from gensim.corpora import WikiCorpus
from gensim.models import word2vec
from gensim.models.word2vec import LineSentence
import multiprocessing
from collections import deque
from opencc import OpenCC
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn import metrics
import warnings

class Wo2Vec(object):
    """
    通过Word2Vec的训练，获取向量值
    """
    def __init__(self):
        """
        构造函数
        """
        self.train_file = "../data/wiki_中文/zhwiki_raw.txt"
        self.train_seg_file = "../data/wiki_中文/zhwiki_seg.txt"
        self.wv_model = None
        # 模型保存地址
        self.save_model_file = "model/wiki.zh.text.model"
        # 词向量保存地址
        self.save_vec_file = "model/wiki.zh.text.vector"

        pass

    def dataprocess(self):
        """
        数据预处理
        :return:
        """
        i = 0
        cc = OpenCC('t2s')
        with open(self.train_file, 'w') as output:
            wiki = WikiCorpus('../data/wiki_中文/zhwiki-latest-pages-articles.xml.bz2', lemmatize=False, dictionary={})
            for text in wiki.get_texts():
                zh_text = ' '.join(text)
                output.write(cc.convert(zh_text) + '\n')
                i += 1
                if i % 10000 == 0:
                    print('成功存储{num}篇文章'.format(num=str(i)))
            print('全部{num}篇文章存储成功'.format(num=str(i)))

    def segFile(self):
        """
        切词
        :return:
        """
        save_fi = open(self.train_seg_file, 'w', encoding='utf-8')
        with open(self.train_file, mode='r', encoding='utf-8', errors='ignore') as InputDoc:
            for line in InputDoc:
                word_list = ' '.join(jieba.cut(line, cut_all=False))
                save_fi.writelines(word_list)
        return save_fi

    def model(self, dataprocess=False, seg=False):
        """
        训练数据
        :param dataprocess 是否做数据预处理
        :param seg 是否分词
        :return:
        """
        if dataprocess:
            self.dataprocess()

        if seg:
            self.segFile()
        logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s', level=logging.INFO)

        self.wv_model = word2vec.Word2Vec(LineSentence(self.train_seg_file),
                          min_count=5, # 词组低于5，不进行向量词组计算
                          size=400, # 向量大小
                          window=3, # 窗口大小
                          workers=multiprocessing.cpu_count())
        self.wv_model.save(self.save_model_file)
        self.wv_model.wv.save_word2vec_format(self.save_vec_file, binary=False)
        return self.wv_model

    def testModel(self):
        """
        验证Word2Vec模型
        :return:
        """
        if self.wv_model is None:
            self.wv_model = word2vec.Word2Vec.load(self.save_model_file)
        # 查找国王最相似的词
        print("计算相似词")
        # DeprecationWarning: Call to deprecated `most_similar` (Method will be removed in 4.0.0, use self.wv.most_similar() instead).
        similar_list = self.wv_model.wv.most_similar('皇上')
        for word in similar_list:
            print(word)
        print("计算相似词，含去除负样本")
        word = self.wv_model.wv.most_similar(positive=[u'皇上', u'国王'], negative=[u'太后'])
        for t in word:
            print(t)
        # 计算两个集合的相似度
        list1 = ['乔峰', '慕容复']
        list2 = ['萧远山', '慕容博']
        print(self.wv_model.wv.n_similarity(list1, list2))

        print("找出不同类的词")
        # 选出集合中不同类的词语
        list3 = ['乔峰', '段誉', '虚竹', '丁春秋']
        print(self.wv_model.wv.doesnt_match(list3))
        print("两个词向量的相近程度")
        print(self.wv_model.wv.similarity(u'书籍', u'书本'))


class SentiAnalysis():
    """
    情感分析
    数据源为某酒店2K条分析数据
    """
    def __init__(self, stop_words_file, save_model_file):
        """

        """
        self.stop_words_file = stop_words_file
        self.stop_words = [w.strip() for w in open(stop_words_file, 'r', encoding='utf-8').readlines()]
        self.wv_model = word2vec.Word2Vec.load(save_model_file)
        self.logger = logging.getLogger()
        logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s', level=logging.INFO)

    def clearText(self, line):
        """
        清洗文本中的特殊字符
        :param line:
        :return:
        """
        ignorechars = string.punctuation + string.digits
        # 去除英文标点和数字
        my_line = str(line).translate(str.maketrans('', '', ignorechars))
        # 去除文本的英文
        temp_line = re.sub("[a-zA-Z0-9]", "", my_line)
        ret_line = re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）]+", "", temp_line)
        return ret_line

    def sen2Word(self, line):
        """
        采用结巴分词，并去除停用词
        :return:
        """

        words = jieba.cut(line, cut_all=False)
        result_words = []
        for word in words:
            # 词不是换行及tab建，并且不在停用词
            if word not in self.stop_words:
                result_words.append(word)

        return ' '.join(result_words)

    def __getVecs(self, wordList):
        """
        建立词向量标准
        :return:
        """
        vecs = []
        for word in wordList:
            try:
                vecs.append(self.wv_model[word])
            except KeyError:
                # 如果没有词向量，则跳过
                continue

        return np.array(vecs, dtype='float')

    def prepareData(self, sourceFile):
        """
        预处理数据，
        A 文本清洗（处理标点及特殊字符）
        B 切词
        C 去停用词
        D 获取词向量（句向量采用加权平均获取）
        :return:
        """
        fi = open(sourceFile, 'r')

        fileVecs = []
        for line in fi.readlines():
            if line.strip() == "":
                continue
            else:
                clear_line = self.clearText(line)

                word_line = self.sen2Word(clear_line)
                self.logger.info("Start line: " + word_line)

                word_vecs = self.__getVecs(word_line.split(' '))
                if len(word_vecs) > 0:
                    vecsArray = sum(np.array(word_vecs)) / len(word_vecs)
                    fileVecs.append(vecsArray)

        return fileVecs

    def createTrasData(self, posInputFile, negInputFile, targetFile):
        """
        构建训练样本
        :param posInputFile 正样本输入数据
        :param negInputFile 负样本输入数据
        :param targetFile 最终汇总数据文件
        :return:
        """
        # 获取了正负样本词向量
        posInput = self.prepareData(posInputFile)
        negInput= self.prepareData(negInputFile)

        # 构建X
        X = posInput[:]
        for neg in negInput:
            X.append(neg)
        X = np.array(X)
        # use 1 for positive sentiment， 0 for negative
        Y = np.concatenate((np.ones(len(posInput)), np.zeros(len(negInput))))
        # write in file
        df_x = pd.DataFrame(X)
        df_y = pd.DataFrame(Y)
        data = pd.concat([df_y, df_x], axis=1)
        # print data
        data.to_csv(targetFile)

        return data

    def train(self, posInputFile, negInputFile, targetFile):
        """
        训练SVM模型，做情感分类
        1 400维向量，先做PCA降维

         :param posInputFile 正样本输入数据
        :param negInputFile 负样本输入数据
        :param targetFile 最终汇总数据文件
        :return:
        """
        # fdir = ''
        # df = pd.read_csv(targetFile)
        # y = df.iloc[:, 1]
        # x = df.iloc[:, 2:]
        data = self.createTrasData(posInputFile, negInputFile, targetFile)
        df = pd.read_csv(targetFile)
        y = df.iloc[:, 1]
        x = df.iloc[:, 2:]

        n_components = 400
        pca = PCA(n_components=n_components)
        pca.fit(x)

        # PCA 作图
        # 根据图显示，前100维向量作为训练主要的数据
        plt.figure(1, figsize=(4, 3))
        plt.clf()
        plt.axes([.2, .2, .7, .7])
        plt.plot(pca.explained_variance_, linewidth=2)
        plt.axis('tight')
        plt.xlabel('n_components')
        plt.ylabel('explained_variance_')
        plt.show()

        # 根据图形取100维
        x_pca = PCA(n_components=100).fit_transform(x)

        clf = svm.SVC(C=2,# 分类数目
                      probability=True)
        clf.fit(x_pca, y)
        print('Test Accuracy: %.2f' % clf.score(x_pca, y))

        # Create ROC curve
        # predict_proba返回的是一个 n 行 k 列的数组， 第 i 行 第 j 列上的数值是模型预测 第 i 个预测样本为某个标签的概率，并且每一行的概率和为1。
        pred_probas = clf.predict_proba(x_pca)[:, 1]  # score
        fpr, tpr, _ = metrics.roc_curve(y, pred_probas)
        roc_auc = metrics.auc(fpr, tpr)

        plt.plot(fpr, tpr, label='area = %.2f' % roc_auc)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.legend(loc='lower right')
        plt.show()


if __name__ == '__main__':
    out = Wo2Vec()
    # 训练前的数据处理
    # out.dataprocess()
    # 切词
    # out.segFile()
    # 训练word2Vec模型
    # out.model()
    out.testModel()
    # senti = SentiAnalysis("../data/stopwords-master/stopWord.txt", "model/wiki.zh.text.model")
    # senti.train("../data/ChnSentiCorp_htl_ba_2000/2000_pos_ut8.txt", "../data/ChnSentiCorp_htl_ba_2000/2000_neg_ut8.txt", "../data/ChnSentiCorp_htl_ba_2000/2000_data.csv")