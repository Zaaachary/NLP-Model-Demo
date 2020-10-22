#! -*- encoding:utf-8 -*-
"""
@File    :   TrainModel.py
@Author  :   Zachary Li
@Contact :   li_zaaachary@163.com
@Dscpt   :   
"""
import torch

from models import Plain_RNN, Criterion
from PrepareData import *


def train(model, data, loss_fn, optimizer, num_epochs=11):
    for epoch in range(num_epochs):
        model.train() # 训练模式
        total_num_words = total_loss = 0.
        for it, (mb_x, mb_x_len, mb_y, mb_y_len) in enumerate(data):
            mb_x = torch.from_numpy(mb_x).to(device).long()
            mb_x_len = torch.from_numpy(mb_x_len).to(device).long()
            mb_input = torch.from_numpy(mb_y[:, :-1]).to(device).long() # EOS之前
            mb_output = torch.from_numpy(mb_y[:, 1:]).to(device).long() # BOS之后
            mb_y_len = torch.from_numpy(mb_y_len - 1).to(device).long()
            mb_y_len[mb_y_len <= 0] = 1
            mb_pred, attn = model(mb_x, mb_x_len, mb_input, mb_y_len)
            # [mb_y_len.max()]->[1, mb_y_len.max()]
            mb_out_mask = torch.arange(mb_y_len.max().item(), device=device)[None, :] < mb_y_len[:, None]
            mb_out_mask = mb_out_mask.float()
            # (pre, target, mask)
            # mb_output是句子单词的索引
            loss = loss_fn(mb_pred, mb_output, mb_out_mask)
            num_words = torch.sum(mb_y_len).item()
            total_loss += loss.item() * num_words
            total_num_words += num_words
            # 更新模型
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
            if it % 100 == 0:
                print("Epoch: ", epoch, 'iteration', it, 'loss:', loss.item())
                # print("Epoch", epoch, "Training loss", total_loss / total_num_words)
        if epoch % 5 == 0:
            evaluate(model, dev_data)
            torch.save(model.state_dict(), './checkpoint/Plain_RNN2.pt')

def evaluate(model, data):
    model.eval()
    total_num_words = total_loss = 0.
    with torch.no_grad():
        for (mb_x, mb_x_len, mb_y, mb_y_len) in data:
            mb_x = torch.from_numpy(mb_x).to(device).long()
            mb_x_len = torch.from_numpy(mb_x_len).to(device).long()
            mb_input = torch.from_numpy(mb_y[:, :-1]).to(device).long()
            mb_output = torch.from_numpy(mb_y[:, 1:]).to(device).long()
            mb_y_len = torch.from_numpy(mb_y_len-1).to(device).long()
            mb_y_len[mb_y_len<=0] = 1

            mb_pred, _ = model(mb_x, mb_x_len, mb_input, mb_y_len)

            mb_out_mask = torch.arange(mb_y_len.max().item(), device=device)[None, :] < mb_y_len[:, None]
            mb_out_mask = mb_out_mask.float()

            loss = loss_fn(mb_pred, mb_output, mb_out_mask)

        num_words = torch.sum(mb_y_len).item()
        total_loss += loss.item() * num_words
        total_num_words += num_words
        print("Evaluation loss", total_loss / total_num_words)


if __name__ == "__main__":
    # ====== DATA PREPARE ===== #
    # load word data
    train_en_w, train_cn_w = load_data(train_file)
    dev_en_w, dev_cn_w = load_data(dev_file)
    # build vocabulary
    en2idx, idx2en = build_vocab(train_en_w)
    cn2idx, idx2cn = build_vocab(train_cn_w)
    # trans word sentences to idx sentences
    train_en, train_cn = sentences2idx(train_en_w, train_cn_w, en2idx, cn2idx, len_sort=False)
    dev_en, dev_cn = sentences2idx(dev_en_w, dev_cn_w, en2idx, cn2idx, len_sort=False)
    # generate batches
    batch_size = 128
    train_data = gen_examples(train_en, train_cn, batch_size)
    dev_data = gen_examples(dev_en, dev_cn, batch_size)

    # ===== TRAIN MODEL ===== #
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dropout, hidden_size = 0.2, 100
    # define model
    encoder = Plain_RNN.PlainEncoder(len(en2idx), hidden_size, 2, dropout)
    decoder = Plain_RNN.PlainDecoder(len(en2idx), hidden_size, 2, dropout)
    # encoder = Plain_RNN.PlainEncoder(len(en2idx), hidden_size, dropout)
    # decoder = Plain_RNN.PlainDecoder(len(en2idx), hidden_size, dropout)
    model = Plain_RNN.PlainSeq2Seq(encoder, decoder)
    model = model.to(device)
    # define loss_function and optimizer
    loss_fn = Criterion.LanguageModelCriterion()
    optimizer = torch.optim.Adam(model.parameters())
    
    train(model, train_data, loss_fn, optimizer, num_epochs=100)