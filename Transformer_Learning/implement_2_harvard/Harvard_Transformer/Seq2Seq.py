#! -*- encoding:utf-8 -*-
"""
@File    :   The_Annotated_Transformer.py
@Author  :   Harvardnlp
@Dscpt   :   CODE from Harvardnlp.
@Link    :   http://nlp.seas.harvard.edu/2018/04/03/attention.html
"""
import math
import copy
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd.Variable as Variable


class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture.
    Base for this and many other models.
    """
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        '''
        Take in(接受) and process masked src and target sequences.
        '''
        memory = self.encode(src, src_mask)
        return self.decode(memory, src_mask, tgt, tgt_mask)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)


class Generator(nn.Module):
    """
    Define standard linear + softmax generation step.
    """
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)
    
    def forward(self, x):
        # x -> Linear(d_model -> vocabulary) -> softmax
        return F.log_softmax(self.proj(x), dim=-1)


