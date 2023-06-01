# -*- coding: utf-8 -*-

import threading
import torch
from BiLSTM_CRF import BiLSTM_CRF
import torch.optim as optim
import time
import numpy as np
import torch.utils.data as data
import sys
import util
import os
import config
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


class Train():
    def __init__(self):
        """
        训练模型
        """
        pass

    def train(self, model, epoch, optimizer, rmrb_loader):
        """
        训练模型
        :param model:
        :param epoch:
        :param optimizer: 优化器
        :param rmrb_loader:
        :return:
        """
        start_time = time.time()
        model.train()  # Turn on the train mode
        total_loss = 0.

        for i in range(len(rmrb_loader)):
            sentence_in, targets = rmrb_loader.iloc[i]
            if sentence_in.shape[0] < 4:
                continue
            # Step 1. Remember that Pytorch accumulates gradients.
            # We need to clear them out before each instance
            model.zero_grad()
            # Step 3. Run our forward pass.
            loss = model.neg_log_likelihood(sentence_in, targets)
            # Step 4. Compute the loss, gradients, and update the parameters by
            # # calling optimizer.step()
            # loss.backward()
            # optimizer.step()
            total_loss += loss.item() / len(sentence_in)

            log_interval = 20
            if i % log_interval == 0 and i != 0:
                cur_loss = total_loss / log_interval
                elapsed = time.time() - start_time
                print('| epoch {:3d} | '
                      'lr {:02.6f} | ms/batch {:5.2f} | '
                      'loss {:5.5f} |'.format(epoch, config.lr,
                                              elapsed * 1000 / log_interval,
                                              cur_loss))
                total_loss = 0
                start_time = time.time()

            # calling optimizer.step()
            loss.backward()
            optimizer.step()

    def evaluate(self, eval_model, data_source):
        """
        评估
        :param eval_model:
        :param data_source:
        :return:
        """
        eval_model.eval()  # Turn on the evaluation mode
        total_loss = 0.
        with torch.no_grad():
            for j in range(len(data_source)):
                sentence_k, targets_k = data_source.iloc[j]
                if sentence_k.shape[0] < 4:
                    continue
                loss_test = eval_model.neg_log_likelihood(sentence_k, targets_k)
                total_loss += float(loss_test) / len(sentence_k)
        return total_loss

    import numpy as np


def trainning():
    """
    训练模型
    :return:
    """
    train_obj = Train()

    training_data, word_to_ix, tag_to_ix = util.data_prepare(config.TRAIN_DATA_PATH)
    training_data.char = training_data.char.apply(lambda c: util.prepare_sequence(c, word_to_ix))
    training_data.tag = training_data.tag.apply(lambda t: torch.tensor([tag_to_ix[t_] for t_ in t], dtype=torch.long))

    model = BiLSTM_CRF(len(word_to_ix), tag_to_ix, config.EMBEDDING_DIM, config.HIDDEN_DIM)
    # 定义优化器为Adam优化器
    optimizer = optim.Adam(model.parameters(), lr=config.lr)

    best_val_loss = float("inf")
    best_model = None

    # Make sure prepare_sequence from earlier in the LSTM section is loaded
    for epoch in range(config.EPOCHES):  # again, normally you would NOT do 300 epochs, it is toy data
        rmrb_loader = training_data.iloc[:-10000].sample(config.BATCH_SIZE)
        rmrb_loader_test = training_data.iloc[-10000:].sample(1000)

        epoch_start_time = time.time()
        train_obj.train(model, epoch, optimizer, rmrb_loader)
        val_loss = train_obj.evaluate(model, rmrb_loader_test)
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '.
              format(epoch, (time.time() - epoch_start_time), val_loss))
        print('-' * 89)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model

        if epoch % 20 == 0:
            torch.save(best_model.state_dict(), config.MODEL_PATH)


if __name__ == '__main__':
    thread = threading.Thread(target=trainning())
    thread.start()