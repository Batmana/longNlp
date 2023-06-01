# -*- coding: utf-8 -*-

import torch
import pandas as pd
import os
import re
import config

def data_prepare(data):
    """
    数据训练准备
    :param data:  数据
    :return:
    """
    # training data
    df = pd.read_csv(data, index_col=0)
    df.char = df['char'].apply(lambda x: eval(x))
    df.tag = df['tag'].apply(lambda y: eval(y))

    word_to_ix = {}
    for i in range(len(df)):
        sentence, _ = df.iloc[i]
        for word in sentence:
            if word not in word_to_ix:
                word_to_ix[word] = len(word_to_ix)

    tag_to_ix = {
        config.START_TAG: 0,
        config.STOP_TAG: 1,
        config.UNK_TAG: 2,
        'O': 3
    }
    tag_to_ix_tmp = dict(
        zip([a + b for a in ['B', 'M', 'E'] for b in ['_TIME', '_PERSON', '_LOCATION', '_ORGANIZATION']], range(4, 16)))

    tag_to_ix.update(tag_to_ix_tmp)

    return df, word_to_ix, tag_to_ix


def argmax(vec):
    """

    :param vec
    :return:
    """
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return idx.item()


def prepare_sequence(seq, to_ix):
    """

    :param seq:
    :param to_ix:
    :return:
    """
    idxs = []
    for w in seq:
        if w not in to_ix.keys():
            w = UNK_TAG
        idxs.append(to_ix[w])
    return torch.tensor(idxs, dtype=torch.long)


def log_sum_exp(vec):
    """
    Compute log sum exp in a numerically stable way for the forward algorithm
    :param vec:
    :return:
    """
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + \
        torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))


def tag2word(sent, tagLst):
    """
    # for eval
    :param tagLst:
    :return:
    """
    res = []
    for i,t in enumerate(tagLst):
        flag = t.split('_')
        if len(flag) > 1:
            if flag[0] == "E":
                tagLst[i] = sent[i] + tagLst[i] + "|"
            else:
                tagLst[i] = sent[i] + tagLst[i]
        else:
            tagLst[i] = '|'
    tmp = "".join(tagLst)
    tmp = set(tmp.split("|"))

    for w in tmp:
        if w:
            w_w = re.findall("[0-9]?[\u4e00-\u9fa5]?",w)
            w_w = "".join(w_w)
            w_t = w.split("_")[-1]
        else:
            continue
        res.append(w_w + " " + w_t)

    return res


def strQ2B(ustring):
    """
    全角转半角
    """
    rstring = ""
    for uchar in ustring:
        inside_code = ord(uchar)
        if inside_code == 12288:                              #全角空格直接转换
            inside_code = 32
        elif inside_code >= 65281 and inside_code <= 65374: #全角字符（除空格）根据关系转化
            inside_code -= 65248
        rstring += chr(inside_code)
    return rstring


def cleanSent(sent):
    """
    清洗句子
    :param sent: 句子
    :return:
    """
    sent = re.sub(r"\n", ",", sent)
    sent = re.sub(r"\t", "", sent)
    return sent


def cut_sentence(data):
    """
    分局
    :param data:
    :return:
    """
    sentLst = data.split("。")
    return sentLst