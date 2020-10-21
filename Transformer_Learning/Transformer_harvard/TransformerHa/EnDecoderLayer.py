#! -*- encoding:utf-8 -*-
"""
@File    :   EncoderDecoder.py
@Author  :   Harvardnlp
@Link    :   http://nlp.seas.harvard.edu/2018/04/03/attention.html
@Dscpt   :   Decoder of Transformer
"""

import copy

import numpy as np
import torch
import torch.nn as nn

from .PublicModules import *


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
