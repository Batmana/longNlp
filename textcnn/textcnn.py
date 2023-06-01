import torch
import torch.nn as nn
import torch.nn.functional as F
import os, sys
import numpy as np
import argparse
import torchtext.legacy.data as data
from torchtext.vocab import Vectors
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

class TextCNN(nn.Module):
    """
    TextCNN的模型
    """
    def __init__(self, args):
        """
        初始化函数
        :param args:
        """
        super().__init__()
        self.args = args
        # 已知词的数量
        Vocab = args.embed_num
        # 每个词向量的长度
        Dim   = args.embed_dim
        # 类别数
        Cla = args.class_num
        # 输出的channel数
        Ci = 1
        # 每种卷积核的数量
        Knum = args.kernel_num
        # 卷积核List，形如[2,3,4]
        Ks = args.kernel_size
        # 词向量, 这里直接随机
        self.embed = nn.Embedding(Vocab, Dim)
        # 卷积层, 卷积核为 K*词向量长度
        self.convs = nn.ModuleList([nn.Conv2d(Ci, Knum, (K, Dim)) for K in Ks])
        self.dropout = nn.Dropout(args.dropout)

        self.fc = nn.Linear(len(Ks)*Knum, Cla)

    def forward(self, x):
        """
        前向计算
        :param x:
        :return:
        """
        # (N,W,D)
        x = self.embed(x)
        # (N,Ci,W,D)
        x = x.unsqueeze(1)
        # len(Ks)*(N,Knum,W)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]
        # len(Ks)*(N,Knum)
        x = [F.max_pool1d(line, line.size(2)).squeeze(2) for line in x]
        # (N,Knum*len(Ks))
        x = torch.cat(x, 1)

        x = self.dropout(x)
        logit = self.fc(x)
        return logit


def train(train_iter, dev_iter, model, args):
    """
    训练模型
    :param train_iter:
    :param dev_iter:
    :param model:
    :param args:
    :return:
    """

    if args.cuda:
        model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    steps = 0
    best_acc = 0
    last_step = 0
    model.train()
    for epoch in range(1, args.epochs + 1):
        for batch in train_iter:
            feature, target = batch.text, batch.label

            feature = feature.data.t()
            target = target.data.sub(1)
            if args.cuda:
                feature, target = feature.cuda(), target.cuda()
            optimizer.zero_grad()
            logits = model(feature)

            loss = F.cross_entropy(logits, target)
            loss.backward()
            optimizer.step()
            steps += 1
            if steps % args.log_interval == 0:
                corrects = (torch.max(logits, 1)[1].view(target.size()).data == target.data).sum()
                train_acc = 100.0 * corrects / batch.batch_size
                sys.stdout.write('\rBatch[{}] - loss: {:.6f}  acc: {:.4f}%({}/{})'.format(steps,
                                                                             loss.item(),
                                                                             train_acc,
                                                                             corrects,
                                                                             batch.batch_size))
            if steps % args.test_interval == 0:
                dev_acc = eval(dev_iter, model, args)
                if dev_acc > best_acc:
                    best_acc = dev_acc
                    last_step = steps
                    if args.save_best:
                        print('Saving best model, acc: {:.4f}%\n'.format(best_acc))
                        save(model, args.save_dir, 'best', steps)
                else:
                    if steps - last_step >= args.early_stopping:
                        print('\nearly stop by {} steps, acc: {:.4f}%'.format(args.early_stopping, best_acc))
                        raise KeyboardInterrupt


def eval(data_iter, model, args):
    model.eval()
    corrects, avg_loss = 0, 0
    y_predict = []
    for batch in data_iter:
        feature, target = batch.text, batch.label
        feature = feature.data.t()
        target = target.data.sub(1)
        if args.cuda:
            feature, target = feature.cuda(), target.cuda()
        logits = model(feature)
        loss = F.cross_entropy(logits, target)
        avg_loss += loss.item()

        y_predict = y_predict + torch.max(logits, 1)[1].view(target.size()).data.numpy().tolist()
        corrects += (torch.max(logits, 1)[1].view(target.size()).data == target.data).sum()

    p, r, f1, s = precision_recall_fscore_support(y_true=target.data.numpy().tolist(), y_pred=y_predict)
    size = len(data_iter.dataset)
    avg_loss /= size
    accuracy = 100.0 * corrects / size
    print('\nEvaluation - loss: {:.6f}  acc: {:.4f}%({}/{} precision: {:.4f} recall: {:.4f}) \n'.format(avg_loss,
                                                                       accuracy,
                                                                       corrects,
                                                                       size,
                                                                       np.average(p, weights=s), np.average(r,weights=s)))
    return accuracy


def save(model, save_dir, save_prefix, steps):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    save_prefix = os.path.join(save_dir, save_prefix)
    save_path = '{}_steps_{}.pt'.format(save_prefix, steps)
    torch.save(model.state_dict(), save_path)


parser = argparse.ArgumentParser(description='TextCNN text classifier')

# learning
parser.add_argument('-lr', type=float, default=0.001, help='initial learning rate [default: 0.001]')
parser.add_argument('-epochs', type=int, default=256, help='number of epochs for train [default: 256]')
parser.add_argument('-batch-size', type=int, default=128, help='batch size for training [default: 128]')
parser.add_argument('-log-interval', type=int, default=1, help='how many steps to wait before logging training status [default: 1]')
parser.add_argument('-test-interval', type=int, default=100, help='how many steps to wait before testing [default: 100]')
parser.add_argument('-save-dir', type=str, default='snapshot', help='where to save the snapshot')
parser.add_argument('-early-stopping', type=int, default=1000, help='iteration numbers to stop without performance increasing')
parser.add_argument('-save-best', type=bool, default=True, help='whether to save when get best performance')
# model
parser.add_argument('-dropout', type=float, default=0.5, help='the probability for dropout [default: 0.5]')
parser.add_argument('-max-norm', type=float, default=3.0, help='l2 constraint of parameters [default: 3.0]')
parser.add_argument('-embedding-dim', type=int, default=128, help='number of embedding dimension [default: 128]')
parser.add_argument('-filter-num', type=int, default=100, help='number of each size of filter')
parser.add_argument('-filter-sizes', type=str, default='3,4,5', help='comma-separated filter sizes to use for convolution')
parser.add_argument('-static', type=bool, default=False, help='whether to use static pre-trained word vectors')
parser.add_argument('-non-static', type=bool, default=False, help='whether to fine-tune static pre-trained word vectors')
parser.add_argument('-multichannel', type=bool, default=False, help='whether to use 2 channel of word vectors')
parser.add_argument('-pretrained-name', type=str, default='sgns.zhihu.word', help='filename of pre-trained word vectors')
parser.add_argument('-pretrained-path', type=str, default='pretrained', help='path of pre-trained word vectors')

# device
parser.add_argument('-device', type=int, default=-1, help='device to use for iterate data, -1 mean cpu [default: -1]')

# option
parser.add_argument('-snapshot', type=str, default=None, help='filename of model snapshot [default: None]')
args = parser.parse_args()

print('Loading data...')


def load_word_vectors(model_name, model_path):
    """
    加载词向量模型
    :param model_name:
    :param model_path:
    :return:
    """
    vectors = Vectors(name=model_name, cache=model_path)
    return vectors


def load_dataset(path, text_field, label_field, args, **kwargs):
    """
    加载数据
    :param text_field:
    :param label_field:
    :param args:
    :param kwargs:
    :return:
    """

    def word_cut(text):
        words = text.strip().split(' ')
        return words
    text_field.tokenize = word_cut
    train_dataset, dev_dataset = data.TabularDataset.splits(
        path=path, format='csv', skip_header=True,
        train='train_data.txt', validation='test_data.txt',
        fields=[
            ('label', label_field),
            ('text', text_field)
        ]
    )
    vectors = load_word_vectors("baike_26g_news_13g_novel_229g.txt", "/Users/zhurunlong/Documents/nlp_model/word2Vec/")
    text_field.build_vocab(train_dataset, dev_dataset, vectors=vectors)

    train_iter, dev_iter = data.Iterator.splits(
        (train_dataset, dev_dataset),
        batch_sizes=(args.batch_size, len(dev_dataset)),
        sort_key=lambda x: len(x.text),
        **kwargs)

    return train_iter, dev_iter

import torch
os.environ['KMP_DUPLICATE_LIB_OK']='True'
#  是否把数据转化为小写 默认值
text_field = data.Field(lower=True)
# 是否把数据表示成序列
label_field = data.Field(sequential=False, use_vocab=False)
train_iter, dev_iter = load_dataset("../data/ad", text_field, label_field, args)
args.embed_num = len(text_field.vocab)
args.embed_dim = 128
args.kernel_num = 128
args.kernel_size = [3, 4, 5]
args.class_num = 2
args.cuda = 0
cnn = TextCNN(args)
train(train_iter=train_iter, dev_iter=dev_iter, model=cnn, args=args)
# train.eval(test_iter, cnn, args)

