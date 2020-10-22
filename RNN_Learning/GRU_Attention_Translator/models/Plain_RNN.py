#! -*- encoding:utf-8 -*-
"""
@File    :   Plain_RNN.py
@Author  :   Zachary Li
@Contact :   li_zaaachary@163.com
@Dscpt   :   
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PlainSeq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(PlainSeq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x, x_lengths, y, y_lengths):
        _, hid = self.encoder(x, x_lengths)
        output, hid = self.decoder(y, y_lengths, hid)
        return output, None

    def translate(self, x, x_lengths, y, max_length=20):
        with torch.no_grad():
            _, hid = self.encoder(x, x_lengths)
            # print(torch.mean(hid))

            preds = []
            batch_size = x.shape[0]
            # sample
            for i in range(max_length):
                # output: [batch_size, y_lengths, vocab_size]
                # 训练的时候y是一个句子，一起decoder训练
                # 测试的时候y是个一个词一个词生成的，所以这里的y是传入的第一个单词，这里是bos
                # 同理y_lengths也是1
                output, hid = self.decoder(y=y, y_len=torch.ones(batch_size).long(), hidden=hid)
                #刚开始循环bos作为模型的首个输入单词，后续更新y，下个预测单词的输入是上个输出单词
                # output.shape = torch.Size([1, 1, 3195])
                # hid.shape = torch.Size([1, 1, 100])

                y = output.max(2)[1].view(batch_size, 1)
                # .max(2)在第三个维度上取最大值,返回最大值和对应的位置索引，[1]取出最大值所在的索引
                preds.append(y)
                # preds = [tensor([[5]], device='cuda:0'), tensor([[24]], device='cuda:0'), ... tensor([[4]], device='cuda:0')]
                # torch.cat(preds, 1) = tensor([[ 5, 24, 6, 22, 7, 4, 3, 4, 3, 4]], device='cuda:0')
        return torch.cat(preds, 1), None


class PlainEncoder(nn.Module):
    """
    x -> embedding -> GRU -> h
    """
    # def __init__(self, vocab_size, hidden_size, dropout=0.2):
    def __init__(self, vocab_size, hidden_size, num_layers=2, dropout=0.2):
        super(PlainEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        # self.rnn = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.rnn = nn.GRU(hidden_size, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, x_len):
        '''
        :param x:
        :param x_len:
        return
        out: [batch, seq_len, hidden_zize]
        hidden: [num_layers=1, batch, hidden_size]
        '''
        x_embedded = self.dropout(self.embedding(x))

        # 为了rnn时能取到真实长度的最后状态，先pack_padded_sequence进行处理
        packed_embedded = nn.utils.rnn.pack_padded_sequence(
            x_embedded, x_len.long().cpu().data.numpy(),  # 因为转到 Numpy 所以需要cpu
            batch_first=True, enforce_sorted=False)
        packed_out, hidden = self.rnn(packed_embedded)  # only input x, init_hidden = 0
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True) # 回到padding长度
        output.contiguous()
        hidden = hidden.contiguous()
        return output, hidden


class PlainDecoder(nn.Module):
    # def __init__(self, vocab_size, hidden_size, dropout=0.2):
    def __init__(self, vocab_size, hidden_size, num_layers=2, dropout=0.2):
        super(PlainDecoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.rnn = nn.GRU(hidden_size, hidden_size, num_layers, batch_first=True) # [batch_size, seq_len, hidden_size]
        # self.rnn = nn.GRU(hidden_size, hidden_size, batch_first=True) # [batch_size, seq_len, hidden_size]
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(dropout)
        # 和PlainEncoder的forward过程大致差不多，区别在于hidden_state不是0而是传入的
        # y: 一个batch的每个中文句子编码
        # hid: hidden_state, context vectors

    def forward(self, y, y_len, hidden):
        # [batch_size, y_lengths, embed_size=hidden_size]
        y = self.dropout(self.embedding(y))

        packed_seq = nn.utils.rnn.pack_padded_sequence(
            y, y_len.long().cpu().data.numpy(),   # 因为转到 Numpy 所以需要cpu
            batch_first=True, enforce_sorted=False)
        out, hidden = self.rnn(packed_seq, hidden)
        output, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)

        output = F.log_softmax(self.fc(output), -1) # [batch_size, y_lengths, vocab_size]
        output.contiguous()
        hidden = hidden.contiguous()
        return output, hidden

