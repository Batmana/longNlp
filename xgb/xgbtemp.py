# -*- coding: utf-8 -*-
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

bbc_text_df = pd.read_csv("../data/BBC/bbc-text.csv")
print(bbc_text_df.head())

plt.figure(figsize=(12, 5))
sns.countplot(x=bbc_text_df.category, color='red')
plt.title('BBC text class distribution', fontsize=16)
plt.ylabel('Class Counts', fontsize=16)
plt.xlabel('Class Label', fontsize=16)
plt.xticks(rotation='vertical')
plt.show()

from gensim import utils
import gensim.parsing.preprocessing as gsp

filters = [
           gsp.strip_tags, # 删除多余的html标签
           gsp.strip_punctuation, # 删除标点符号
           gsp.strip_multiple_whitespaces, # 删除重复的空白字符
           gsp.strip_numeric, # 删除数字
           gsp.remove_stopwords, # 删除停用词
           gsp.strip_short, # 删除小于3个的单词
           gsp.stem_text # 转变为小写
          ]

def clean_text(s):
    s = s.lower()
    s = utils.to_unicode(s)
    for f in filters:
        s = f(s)
    return s

print(bbc_text_df.iloc[2, 1])
print(clean_text(bbc_text_df.iloc[2, 1]))

from wordcloud import WordCloud


def plot_word_cloud(text):
    wordcloud_instance = WordCloud(width=800, height=800,
                                   background_color='black',
                                   stopwords=None,
                                   min_font_size=10).generate(text)

    plt.figure(figsize=(8, 8), facecolor=None)
    plt.imshow(wordcloud_instance)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.show()


texts = ''
for index, item in bbc_text_df.iterrows():
    texts = texts + ' ' + clean_text(item['text'])

plot_word_cloud(texts)


def plot_word_cloud_for_category(bbc_text_df, category):
    text_df = bbc_text_df.loc[bbc_text_df['category'] == str(category)]
    texts = ''
    for index, item in text_df.iterrows():
        texts = texts + ' ' + clean_text(item['text'])
    plot_word_cloud(texts)

plot_word_cloud_for_category(bbc_text_df,'tech')

# 对于任何自然语言处理问题，都有必要进行向量空间建模。以两个最常见的向量空间模型为例：Doc2Vec和Tf-Idf
# 把数据分成特征和类别。
df_x = bbc_text_df['text']
df_y = bbc_text_df['category']

# 使用“Genism”库的Doc2Vec编写一般/通用“Doc2VecTransfoemer”
from gensim.models.doc2vec import TaggedDocument, Doc2Vec
from sklearn.base import BaseEstimator
from sklearn import utils as skl_utils
from tqdm import tqdm

import multiprocessing
import numpy as np

class Doc2VecTransformer(BaseEstimator):

    def __init__(self, vector_size=100, learning_rate=0.02, epochs=20):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self._model = None
        self.vector_size = vector_size
        self.workers = multiprocessing.cpu_count() - 1

    def fit(self, df_x, df_y=None):
        tagged_x = [TaggedDocument(clean_text(row).split(), [index]) for index, row in enumerate(df_x)]
        model = Doc2Vec(documents=tagged_x, vector_size=self.vector_size, workers=self.workers)

        for epoch in range(self.epochs):
            model.train(skl_utils.shuffle([x for x in tqdm(tagged_x)]), total_examples=len(tagged_x), epochs=1)
            model.alpha -= self.learning_rate
            model.min_alpha = model.alpha

        self._model = model
        return self

    def transform(self, df_x):
        return np.asmatrix(np.array([self._model.infer_vector(clean_text(row).split())
                                     for index, row in enumerate(df_x)]))

doc2vec_trf = Doc2VecTransformer()
doc2vec_features = doc2vec_trf.fit(df_x).transform(df_x)
print(doc2vec_features)

# 下面以LogisticRegression, RandomForest和XGBoost为例进行操作。

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

# Doc2Vec和RandomForest管道
pl_log_reg = Pipeline(steps=[('doc2vec', Doc2VecTransformer()),
                             ('log_reg', LogisticRegression(multi_class='multinomial', solver='saga', max_iter=100))])
scores = cross_val_score(pl_log_reg, df_x, df_y, cv=5,scoring='accuracy')
print('Accuracy for Logistic Regression: ', scores.mean())

# Doc2Vec和XGBoost管道
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = "TRUE"
import xgboost as xgb

pl_xgb = Pipeline(steps=[('doc2vec', Doc2VecTransformer()),
                         ('xgboost', xgb.XGBClassifier(objective='multi:softmax'))])
scores = cross_val_score(pl_xgb, df_x, df_y, cv=5)
print('Accuracy for XGBoost Classifier : ', scores.mean())


from sklearn.feature_extraction.text import TfidfVectorizer

class Text2TfIdfTransformer(BaseEstimator):

    def __init__(self):
        self._model = TfidfVectorizer()
        pass

    def fit(self, df_x, df_y=None):
        df_x = df_x.apply(lambda x : clean_text(x))
        self._model.fit(df_x)
        return self

    def transform(self, df_x):
        return self._model.transform(df_x)
# Tf-Idf
tfidf_transformer = Text2TfIdfTransformer()
tfidf_vectors = tfidf_transformer.fit(df_x).transform(df_x)
print(tfidf_vectors.shape)
print(tfidf_vectors)

pl_log_reg_tf_idf = Pipeline(steps=[('tfidf', Text2TfIdfTransformer()),
                             ('log_reg', LogisticRegression(multi_class='multinomial', solver='saga', max_iter=100))])
scores = cross_val_score(pl_log_reg_tf_idf, df_x, df_y, cv=5,scoring='accuracy')
print('Accuracy for Tf-Idf & Logistic Regression: ', scores.mean())

# Tf-Idf & RandomForest
from sklearn.ensemble import RandomForestClassifier

pl_random_forest_tf_idf = Pipeline(steps=[('tfidf', Text2TfIdfTransformer()),
                                   ('random_forest', RandomForestClassifier())])
scores = cross_val_score(pl_random_forest_tf_idf, df_x, df_y, cv=5, scoring='accuracy')
print('Accuracy for Tf-Idf & RandomForest : ', scores.mean())

# Tf-Idf & XGBoost
pl_xgb_tf_idf = Pipeline(steps=[('tfidf', Text2TfIdfTransformer()),
                         ('xgboost', xgb.XGBClassifier(objective='multi:softmax'))])
scores = cross_val_score(pl_xgb_tf_idf, df_x, df_y, cv=5)
print('Accuracy for Tf-Idf & XGBoost Classifier : ', scores.mean())