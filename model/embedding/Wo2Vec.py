#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   Wo2Vwc.py
@Time    :   2023/05/30 10:38:26
@Author  :   peter
@Version :   1.0
@Contact :   peterzhu235@gmail.com
@Desc    :   采用pytorch实现Word2Vec
'''
# here put the import lib
import torch, random
import numpy as np
import jieba
import torch.nn as nn
import torch.nn.functional as F
# torch的工具
import torch.utils.data as tud
from configparser import ConfigParser
from collections import Counter
import os

current_path = os.path.dirname(os.getcwd())
home_path = os.path.expanduser('~') + "/zhu/"

random.seed(1)
np.random.seed(1)
torch.manual_seed(1)

# Word2Vec模型
class Wo2Vec(nn.Module):
    def __init__(self, vocab_size, embed_size, device) -> None:
        """定义Word2Vec网络

        Arguments:
            vocab_size -- 词典大小
            embed_size -- 隐藏层embedding长度
        """
        self.device = device
        super(Wo2Vec, self).__init__()
       
        self.vocab_size = int(vocab_size)
        self.embed_size = int(embed_size)
   
        # 假设词典大小为M, 隐藏层大小为N
        self.in_emb = nn.Embedding(num_embeddings= self.vocab_size, embedding_dim=self.embed_size)
        # 输出层embedding
        self.out_emb = nn.Embedding(num_embeddings= self.vocab_size, embedding_dim=self.embed_size)

    # 前向网络, 输入应该为中心词 / 窗口词 / 负采样词
    def forward(self, centor_word, window_word, neg_word):
        """
        前向网络
        Arguments:
            centor_word -- 中心词
            window_word -- 窗口词
            neg_word    -- 负采样词
        """
        # 计算中心词向量 [batch_size, embed_dim]
        centor_word_emb = self.in_emb(centor_word).to(self.device)
        # 计算窗口词向量 [batch_size, (window * 2), embed_dim]
        window_word_emb = self.out_emb(window_word).to(self.device)
        # 负采样词向量[batch_size, (window * 2 * k), embed_dim]
        neg_word_emb = self.out_emb(neg_word).to(self.device)

        # 前向计算
        # 将中心词的表示扩展为需要的维度
        centor_word_emb = centor_word_emb.unsqueeze(2)

        # 矩阵相乘，得到隐藏层表示
        # [batch_size, embed_dim] * [batch_size, (window * 2), embed_dim] = [batch_size, (window * 2), 1]
        # 参数window_word_emb 和 centor_word_emb 不能变化位置
        pos_dot = torch.bmm(window_word_emb, centor_word_emb)
        # 得到正样本的每个样本的得分[batch_size, (window * 2)]
        pos_dot = pos_dot.squeeze(2)

        # 负样本同样的操作
        # 注意负号，参考公式可以知负样本的概率越小越好，所以位负号
        # [batch_size, (window * 2 * K), 1]
        neg_dot = torch.bmm(neg_word_emb, -centor_word_emb).to(self.device)
        neg_dot.squeeze(2)

        # Softmax
        # 两分类模型，softmax为求正样本的score + 负样本的score
        log_pos = F.logsigmoid(pos_dot).sum(1)
        log_neg = F.logsigmoid(neg_dot).sum(1)

        loss = log_pos + log_neg

        return -loss
        
# 数据操作类
class Wo2VecDataSet(tud.Dataset):
    """数据操作类

    Arguments:
        tud -- _description_
    """
    def __init__(self, text, word2idx, word_freqs) -> None:
        """_summary_

        Arguments:
            text -- _description_
            word2idx -- _description_
            word_freqs -- _description_
        """
        global current_path
        super(Wo2VecDataSet, self).__init__()
        # 注意下面重写的方法
        # 将语句转换为word编码 
        self.text_encoded = [word2idx.get(word, word2idx['<UNK>']) for word in text]
         # nn.Embedding需要传入LongTensor类型
        self.text_encoded = torch.LongTensor(self.text_encoded)
        self.word2idx = word2idx
        self.word_freqs = torch.Tensor(word_freqs)
        # 窗口大小
        config = ConfigParser()
        config.read(current_path + "/config/word2vec.ini")
      
        self.C = int(config.get("model", "C"))
        self.K = int(config.get("model", "K"))

    def __len__(self):
        """获取长度信息

        Returns:
            _type_: _description_
        """
        return len(self.text_encoded)
    # 重写方法，
    def __getitem__(self, idx):
        """这个function用于返回：中心词（center_words），周围词（pos_words），负采样词（neg_words）

        Args:
            idx (_type_):中心词索引

        Returns:
            Any: _description_
        """
        # 取出中心词
        center_words = self.text_encoded[idx]
        # 取出所有周围词索引，用于下面的tensor(list)操作
        pos_indices = list(range(idx - self.C, idx)) + list(range(idx + 1, idx + self.C + 1))
        # 为了避免索引越界，所以进行取余处理，如：设总词数为100，则[1]取余为[1]，而[101]取余为[1]
        pos_indices = [i %  len(self.text_encoded) for i in pos_indices]
        # tensor(lsit)操作，取出所有周围词
        pos_words = self.text_encoded[pos_indices]
        # 负样本词
        # torch.multinomial作用是对self.word_freqs做K * pos_words.shape[0]次取值，输出的是self.word_freqs对应的下标
        # 取样方式采用有放回的采样，并且self.word_freqs数值越大，取样概率越大
        # 实际上从这里就可以看出，这里用的是skip-gram方法，并且采用负采样（Negative Sampling）进行优化
        neg_words = torch.multinomial(self.word_freqs, self.K * pos_words.shape[0], True)
        
    
        # while 循环是为了保证 neg_words中不能包含周围词
        # Angel Hair：实际上不需要这么处理，因为我们遇到的都是非常大的数据，会导致取到周围词的概率非常非常小，
        # 这里之所以这么做是因为本文和参考文所提供的数据太小，导致这个概率变大了，会影响模型
        while len(set(pos_indices) & set(neg_words.numpy().tolist())) > 0:
             neg_words = torch.multinomial(self.word_freqs, self.K * pos_words.shape[0], True)
      
        return center_words, pos_words, neg_words

# 模型使用
class Wo2VecPipe():
    def __init__(self, device) -> None:
        # 窗口大小
        global current_path
        config = ConfigParser()
        config.read(current_path + "/config/word2vec.ini")

        self.MAX_VOCAB_SIZE = int(config.get("model", "MAX_VOCAB_SIZE")) 
        self.EMBEDDING_SIZE = int(config.get("model", "EMBEDDING_SIZE"))
        self.batch_size = int(config.get("model", "batch_size"))
        self.epochs = int(config.get("model", "epochs"))
        self.embedding_weights = None
        self.device = device
        return
    

    def train(self, text, word2idx, word_freqs):
        """
        训练数据
        """
        model = Wo2Vec(self.MAX_VOCAB_SIZE, self.EMBEDDING_SIZE, self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        self.dataset = Wo2VecDataSet(text, word2idx, word_freqs)
        dataloader = tud.DataLoader(self.dataset, self.batch_size, shuffle=True)
       
        for e in range(self.epochs):
            for i, (input_labels, pos_labels, neg_labels) in  enumerate(dataloader):
                input_labels.long()
                pos_labels.long()
                neg_labels.long()

                optimizer.zero_grad()
                #.mean()默认不设置dim的时候，返回的是所有元素的平均值
                loss = model(input_labels, pos_labels, neg_labels).mean()
                loss.backward()
                optimizer.step()

                if i % 100 == 0:
                    print('epoch', e, 'iteration', i, loss.item())

        self.embedding_weights = model.input_embedding()
        # 保存模型
        if not os.path.exists(home_path + "/.nlp"):
            os.mkdir(home_path + "/.nlp", 755)
        if not os.path.exists(home_path + "/.nlp/word2vec"):
            os.mkdir(home_path + "/.nlp/word2vec", 755)
        torch.save(model.state_dict(), home_path + "/.nlp/word2vec/embedding-{}.th".format(self.EMBEDDING_SIZE))

    def get_vec(self, word):
        """获取词的Embedding

        Args:
            word (_type_): _description_
        """
        device = torch.device('cpu')
        model = Wo2Vec(self.MAX_VOCAB_SIZE, self.EMBEDDING_SIZE)
        if not self.embedding_weights:
            model.load_state_dict(torch.load(home_path + "/.nlp/word2vec/embedding-{}.th".format(self.EMBEDDING_SIZE)), device)

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available()  else "cpu")
    with open("../../data/剑来.txt", "r", encoding='utf-8') as f:
        data = []
        for line in f.readlines():
            strip_line = line.strip()
            # 清除空行
            if strip_line:
                # 分词
                cut_words = list(jieba.cut(strip_line))
                # 标点符号集
                stopwords = '''~!@#$%^&*()_+`1234567890-={}[]:：";'<>,.?/|\、·！（）￥“”‘’《》，。？/—-【】….'''
                stopwords_set = set([i for i in stopwords])
                stopwords_set.add("br") # 异常词也加入此集，方便去除
                for s in stopwords_set:
                    cut_line = " ".join(cut_words).replace(s, "")
                cut_line.replace("   "," ").replace("  "," ")
                data.append(cut_line)
        text = " ".join(data)
        text = text.lower().split() #　分割成单词列表
     
        vocab_dict = dict(Counter(text).most_common(100 - 1)) # 得到单词字典表，key是单词，value是次数
        vocab_dict['<UNK>'] = len(text) - np.sum(list(vocab_dict.values())) # 把不常用的单词都编码为"<UNK>"

        # 构建词值对
        word2idx = {word:i for i, word in enumerate(vocab_dict.keys())}
        idx2word = {i:word for i, word in enumerate(vocab_dict.keys())}

        # 计算和处理频率
        word_counts = np.array(list(vocab_dict.values()), dtype=np.float32)
        word_freqs = (word_counts / np.sum(word_counts))** (3./4.) # 所有的频率为原来的 0.75 次方， 论文中的推荐方法，由图像分析可推测这样可以一定程度上提高频率低的权重，降低频率高的权重

        model = Wo2VecPipe(device)
        model.train(text, word2idx, word_freqs)
