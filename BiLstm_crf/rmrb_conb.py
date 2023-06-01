# -*- coding: utf-8 -*-
import re
import pandas as pd
import util


# 将人工标注转化为BMEWO标注
def BMEWO(sent):
    w_t_dict = {'t': '_TIME',
                'nr': '_PERSON',
                'ns': '_LOCATION',
                'nt': '_ORGANIZATION'
    }
    charLst = []
    tagLSt = []

    sent_split_lst = sent.strip().split("  ")
    for w in sent_split_lst:
        if "/" in w:
            ww = w.split("/")[0]
            wtag = w.split("/")[1]
            ww2char = [c for c in ww]
            charLst.extend(ww2char)
            # 获得词的长度
            wlen = len(ww)
            # 初始化标注
            # 其他标注成'O'
            BMEWO_tag = ['O' for _ in ww]

            # 如果是['t','nr','ns','nt'] 标注
            if wtag in ['t', 'nr', 'ns', 'nt']:
                BMEWO_tag[0] = 'B' + w_t_dict[wtag]
                BMEWO_tag[-1] = 'E' + w_t_dict[wtag]
                BMEWO_tag[1:-1] = ['M' + w_t_dict[wtag] for _ in BMEWO_tag[1:-1]]
            tagLSt.extend(BMEWO_tag)
        else:
            continue
    return charLst, tagLSt


def deal_(text_line):
    """
    数据处理
    :param text_line:
    :return:
    """
    # 合并中括号里面的内容
    blacketLst = re.findall("\[.*?\]", text_line)
    if blacketLst:
        for b in blacketLst:
            b_trans = "".join([i.split("/")[0] for i in b[1:-1].split("  ")]) + "/"
            text_line = text_line.replace(b, b_trans)
    # 将姓名空格去除
    text_line = re.sub("(\w+)/nr  (\w+)/nr", r"\1\2/nr", text_line)
    # 将时间空格去除
    text_line = re.sub("(\w+)/t  (\w+)/t", r"\1\2/t", text_line)
    text_line = re.sub("(\w+)/t  (\w+)/t  (\w+)/t", r"\1\2\3/t", text_line)
    text_line = re.sub("(\w+)/t  (\w+)/t  (\w+)/t  (\w+)/t", r"\1\2\3\4/t", text_line)

    # 去除（/w ）/w
    text_line = re.sub("（/w  ","",text_line)
    text_line = re.sub("  ）/w", "", text_line)
    return text_line


def dealData(filename):
    """
    处理数据
    :return:
    """
    file = open(filename, 'r', encoding="utf8")

    char4train = []
    tag4train = []

    try:
        while True:
            text_line_ = file.readline()
            if text_line_:
                text_line_ = text_line_[23:]
                # 分句
                text_line_cut_Lst = text_line_.split("。/w  ")

                for text_line in text_line_cut_Lst:
                    # 去除最后的标点符号
                    text_line = text_line.strip()
                    if text_line[-2:] == "/w":
                        text_line = text_line[:-3]
                    if len(text_line) > 20:
                        text_line = deal_(text_line)
                        print(text_line)
                        text_line = util.strQ2B(text_line)

                        charLst, tagLSt = BMEWO(text_line)
                        char4train.append(charLst)
                        tag4train.append(tagLSt)

            else:
                break
    finally:
        file.close()
        return char4train, tag4train


if __name__ == "__main__":
    # 数据处理
    """
    该脚本将人民日报数据 语料合并
    主要合并 t ,nr , ns, nt
    1、将分开的姓和名合并
    2、将[]中的大粒度词合并
    3、将时间合并
    4、将全角字符统一转为半角字符
    """

    RMRB_DATA_PATH = "../data/NER_corpus_chinese-master/Peoples_Daily/rmrb1998-01.txt"
    res_df = pd.DataFrame()
    res_df['char'], res_df['tag'] = dealData(RMRB_DATA_PATH)
    res_df.to_csv(r"../data/NER_corpus_chinese-master/Peoples_Daily/rmrb4train.csv", encoding='utf-8')




