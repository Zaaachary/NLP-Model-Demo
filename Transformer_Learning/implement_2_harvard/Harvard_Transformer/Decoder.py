#! -*- encoding:utf-8 -*-
"""
@File    :   Decoder.py
@Author  :   Harvardnlp
@Link    :   http://nlp.seas.harvard.edu/2018/04/03/attention.html
@Dscpt   :   Decoder of Transformer
"""

import copy

import numpy as np
import torch
import torch.nn as nn

from Harvard_Transformer.Basic import *


class Decoder(nn.Module):
    """
    Generic N layer decoder with masking.
    """
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)

class DecoderLayer(nn.Module):
    """
    Decoder is made of self-attn, src-attn, and feed forward (defined below)
    """
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn   # self-attention
        self.src_attn = src_attn     # source-attention over the output of the encoder stack
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)
 
    def forward(self, x, memory, src_mask, tgt_mask):
        '''
        Follow Figure 1 (right) for connections.
        '''
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)


def subsequent_mask(size):
    '''
    Mask out subsequent positions. 
    Words are blocked for attending to future words during training.
    '''
    attn_shape = (1, size, size)
    # np.triu   Upper triangle of an array. 
    # Return a copy of a matrix with the elements below the `k`-th diagonalzeroed.
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0