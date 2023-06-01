# -*- coding: utf-8 -*-
import pandas as pd
import os
import re

START_TAG = "<START>"
STOP_TAG = "<STOP>"
UNK_TAG = "<UNK>"

#
EPOCHES = 300
# 词嵌入向量大小
EMBEDDING_DIM = 300
# 隐层向量大小
HIDDEN_DIM = 300
# 批量大小
BATCH_SIZE = 128
# 学习率
lr = 0.0001

TRAIN_DATA_PATH = r"../data/NER_corpus_chinese-master/Peoples_Daily/rmrb4train.csv"
MODEL_PATH = os.path.join(r'model/',
                          "model_emb_{}_hidden_{}_batch_{}_baseline2".format(
                              EMBEDDING_DIM, HIDDEN_DIM, BATCH_SIZE)
                          )