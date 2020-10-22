#! -*- encoding:utf-8 -*-
"""
@File    :   Criterion.py
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

    def forward(self, origin, result, mask):
        '''
        :param input: batch_size, seq_len, vocab_size
        :param target: batch_size, seq_len
        :param mask: batch_size, seq_len
        '''
        origin = origin.contiguous().view(-1, origin.size(2))  # b,len,vocab -> b*len, vocab
        result = result.contiguous().view(-1, 1)
        mask = mask.contiguous().view(-1, 1)

        loss = -origin.gather(1, result) * mask
        loss = torch.sum(loss) / torch.sum(mask)
        return loss
