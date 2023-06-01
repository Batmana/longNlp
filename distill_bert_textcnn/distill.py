# -*- coding: utf-8 -*-

import torch
from transformers import BertTokenizer
from ptbert import *
from small import *
from utils import *
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from keras.preprocessing.text import Tokenizer

FTensor = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor


def load_data():
    """
    加载数据
    :param name:
    :return:
    """
    tokenizer = Tokenizer(filters='', lower=True, split=' ', oov_token=1)
    x_train, y_train = [], []
    text_train = []
    text_test = []
    for line in open('../data/ad/train_data.txt', encoding="utf-8").read().strip().split('\n'):
        label, text = line.split(',', 1)
        if text.strip() == "text":
            continue
        temp_label = 1
        if label.strip() == "2":
            temp_label = 0
        text_train.append(text.strip())
        x_train.append(text.strip().split(' '))
        y_train.append(int(temp_label))

    x_test, y_test = [], []
    for line in open('../data/ad/test_data.txt', encoding="utf-8").read().strip().split('\n'):
        label, text = line.split(',', 1)
        if text.strip() == "text":
            continue
        temp_label = 1
        if label.strip() == "2":
            temp_label = 0
        text_test.append(text.strip())
        x_test.append(text.strip().split(' '))
        y_test.append(int(temp_label))

    tokenizer.fit_on_texts(x_train + x_test)
    x_train = tokenizer.texts_to_sequences(x_train)
    x_test = tokenizer.texts_to_sequences(x_test)

    v_size = len(tokenizer.word_index) + 1

    return (x_train, y_train, text_train), \
           (x_test, y_test, text_test), \
           v_size

def train(x_tr, t_tr, b_size):
    """
    训练数据
    :param x_tr: 训练数据
    :param t_tr: 分词前数据
    :param b_size batch_size
    :return:
    """
    l_tr = list(map(lambda x: min(len(x), x_len), x_tr))
    x_tr = sequence.pad_sequences(x_tr, maxlen=x_len)

    model = CNN(v_size, 256, 128, 2)
    if USE_CUDA:
        model = model.cuda()

    with torch.no_grad():
        t_tr = np.vstack([teacher.predict(text) for text in tqdm(t_tr)])

    opt = optim.Adam(model.parameters(), lr=lr)
    ce_loss = nn.NLLLoss()
    mse_loss = nn.MSELoss()
    for epoch in range(epochs):
        losses = []
        model.train()
        for i in range(0, len(x_tr), b_size):
            model.zero_grad()
            # 输入变量
            bx = Variable(LTensor(x_tr[i:i + b_size]))
            # 输出label
            by = Variable(LTensor(y_tr[i:i + b_size]))
            # 句长度
            bl = Variable(LTensor(l_tr[i:i + b_size]))
            # Tearch - softmax输出的概率
            bt = Variable(FTensor(t_tr[i:i + b_size]))
            # softmax，log_softmax
            py1, py2 = model(bx, bl)
            loss = alpha * ce_loss(py2, by) + (1 - alpha) * mse_loss(py1, bt)  # in paper, only mse is used
            loss.backward()
            opt.step()
            losses.append(loss.item())
    return model


def eval(x_test, y_test, model):
    """
    训练数据
    :param x_test:
    :return:
    """
    x_te = sequence.pad_sequences(x_test, maxlen=x_len)
    l_te = list(map(lambda x: min(len(x), x_len), x_te))
    pre_y = []
    for i in range(0, len(x_te), len(x_te)):
        model.zero_grad()
        # 输入变量
        bx = Variable(LTensor(x_te[i:i + len(x_te)]))
        # 句长度
        bl = Variable(LTensor(l_te[i:i + len(x_te)]))
        # softmax，log_softmax
        py1, py2 = model(bx, bl)
        pre_y = torch.max(py1, 1)[1].data.numpy().tolist()
    print(y_test)
    print(pre_y)
    p, r, f1, s = precision_recall_fscore_support(pre_y, y_test)

    print("pre", p, "recall", r, "f1", f1)


class Teacher(object):
    """
    教师模型：Bert
    """
    def __init__(self, bert_model='bert-base-chinese', max_seq=128):
        self.max_seq = max_seq
        self.tokenizer = BertTokenizer.from_pretrained(bert_model, do_lower_case=True)
        self.model = torch.load('cache/model')
        self.model.eval()

    def predict(self, text):
        """
        预测，输出softmax
        :param text:
        :return:
        """
        tokens = self.tokenizer.tokenize(text)[:self.max_seq]
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)
        padding = [0] * (self.max_seq - len(input_ids))
        input_ids = torch.tensor([input_ids + padding], dtype=torch.long).to(device)
        input_mask = torch.tensor([input_mask + padding], dtype=torch.long).to(device)
        logits = self.model(input_ids, input_mask, None)
        return F.softmax(logits, dim=1).detach().cpu().numpy()

class CNN(nn.Module):
    def __init__(self, x_dim, e_dim, h_dim, o_dim):
        super(CNN, self).__init__()
        self.emb = nn.Embedding(x_dim, e_dim, padding_idx=0)
        self.dropout = nn.Dropout(0.2)
        self.conv1 = nn.Conv2d(1, h_dim, (3, e_dim))
        self.conv2 = nn.Conv2d(1, h_dim, (4, e_dim))
        self.conv3 = nn.Conv2d(1, h_dim, (5, e_dim))
        self.fc = nn.Linear(h_dim * 3, o_dim)
        self.softmax = nn.Softmax(dim=1)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x, lens):
        embed = self.dropout(self.emb(x)).unsqueeze(1)
        c1 = torch.relu(self.conv1(embed).squeeze(3))
        p1 = torch.max_pool1d(c1, c1.size()[2]).squeeze(2)
        c2 = torch.relu(self.conv2(embed).squeeze(3))
        p2 = torch.max_pool1d(c2, c2.size()[2]).squeeze(2)
        c3 = torch.relu(self.conv3(embed).squeeze(3))
        p3 = torch.max_pool1d(c3, c3.size()[2]).squeeze(2)
        pool = self.dropout(torch.cat((p1, p2, p3), 1))
        hidden = self.fc(pool)
        return self.softmax(hidden), self.log_softmax(hidden)

if __name__ == '__main__':
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    teacher = Teacher()

    import pickle
    from tqdm import tqdm

    x_len = 50
    b_size = 64
    lr = 0.002
    epochs = 10
    alpha = 0.5     # portion of the original one-hot CE loss

    n_iter = 5
    p_mask = 0.1
    p_ng = 0.25
    ngram_range = (3, 6)
    teach_on_dev = True

    (x_tr, y_tr, t_tr), (x_te, y_te, t_ter), v_size = load_data()
    cnn_model = train(x_tr, t_tr, b_size)
    eval(x_te, y_te, cnn_model)
    # l_te = list(map(lambda x: min(len(x), x_len), x_te))
    # x_te = sequence.pad_sequences(x_te, maxlen=x_len)



