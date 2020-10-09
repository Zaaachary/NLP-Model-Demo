#! -*- encoding:utf-8 -*-
"""
@File    :   Encoder.py
@Author  :   Harvardnlp
@Link    :   http://nlp.seas.harvard.edu/2018/04/03/attention.html
@Dscpt   :   Encoder of Transformer
"""

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from Harvard_Transformer.Basic import *

class Encoder(nn.Module):
    """
    Core encoder is a stack of N layers.
    i.e. Encoder = N * Encoder_layers + LayerNorm
    """
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        # self.norm = nn.LayerNorm(d_model, eps=1e-6)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        '''
        Pass the x and mask through each layer in turn, and add layerNorm
        '''
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class EncoderLayer(nn.Module):
    """
    Encoder is made up of [self-attn] and [feed forward]
    """
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)



