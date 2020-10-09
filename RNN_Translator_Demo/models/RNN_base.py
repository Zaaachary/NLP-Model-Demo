#! -*- encoding:utf-8 -*-
"""
@File    :   RNN_base.py
@Author  :   Zachary Li
@Contact :   li_zaaachary@163.com
@Dscpt   :   
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class LanguageModelCriterion(nn.Module):
    """
    masked cross entropy loss
    """
    def __init__(self):
        super(LanguageModelCriterion, self).__init__()

    def forward(self, inp, target, mask):
        '''
        :param input: batch_size, seq_len, vocab_size
        :param target: batch_size, seq_len
        :param mask: batch_size, seq_len
        '''
        inp = inp.contiguous().view(-1, inp.size(2)) # b,len,vocab -> b*len, vocab
        target = target.contiguous().view(-1, 1)
        mask = mask.contiguous().view(-1, 1)
        output = -inp.gather(1, target) * mask
        output = torch.sum(output) / torch.sum(mask)
        return output


class PlainEncoder(nn.Module):
    """
    x -> embedding -> GRU
    """
    def __init__(self, vocab_size, hidden_size, dropout=0.2):
        super(PlainEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.rnn = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, lengths):
        # (排序好元素，排序好元素下标)
        sorted_len, sorted_idx = lengths.sort(0, descending=True) # 把batch里的seq按照长度降序排列
        x_sorted = x[sorted_idx.long()]
        embedded = self.dropout(self.embed(x_sorted))
        # 句子padding到一样长度的（真实句长会比padding的短）
        # 为了rnn时能取到真实长度的最后状态，先pack_padded_sequence进行处理
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, sorted_len.long().cpu().data.numpy(),
        batch_first=True)
        # out:[batch, seq_len, hidden_zize]
        # hidden: [num_layers=1, batch, hidden_size]
        packed_out, hidden = self.rnn(packed_embedded)
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True) # 回到padding长度
        _, original_idx = sorted_idx.sort(0, descending=False) # 排序回原来的样子
        out = out[original_idx.long()].contiguous() # [batch_size, seq_len, hidden_size]
        hidden = hidden[:, original_idx.long()].contiguous() # [num_layers, batch_size, hidden_size]
        # print("out.shape: ", out.shape, 'hidden.shape: ', hidden.shape)
        return out, hidden[[-1]] # hidden[[-1]], 相当于out[:, -1]