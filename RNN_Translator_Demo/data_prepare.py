#! -*- encoding:utf-8 -*-
"""
@File    :   data_prepare.py
@Author  :   Zachary Li
@Contact :   li_zaaachary@163.com
@Dscpt   :   Translator

Reference: https://www.cnblogs.com/douzujun/p/13624567.html
"""
import os
import sys
import math
import random
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import nltk

# nltk.download('punkt')

train_file = './data/train.txt'
dev_file = './data/dev.txt'


def load_data(file_path):
    '''
    load data from file_path
    '''
    cn, en = [], []
    example_num = 0
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            line = line.strip().split('\t')
            en.append(['BOS'] + nltk.word_tokenize(line[0].lower())+['EOS'])
            cn.append(['BOS'] + [word for word in line[1]] +['EOS'])
            example_num += 1
    
    return en, cn

def build_vocab(sentences, max_size=50000):
    '''
    build vocabulary from sentence, remain the top max_size word.
    '''
    word_count = Counter()
    for sentence in sentences:
        for word in sentence:
            word_count[word] += 1
    ls = word_count.most_common(max_size)   # top max_size words

    word2idx = {w[0]:index+2 for index, w in enumerate(ls)}  # {word: index}
    word2idx['UNK'] = 0
    word2idx['PAD'] = 1

    idx2word = {v:k for k,v in word2idx.items()}

    return word2idx, idx2word

def sentences2idx(sentences, word2idx):
    '''
    transform sentcen to idx sequence
    '''
    idx_sentences = [[word2idx.get(word, 0) for word in sentence] for sentence in sentences]
    return idx_sentences

def get_batches(total_num, batch_size, shuffle=True):
    '''
    shuffle and return index of sentences for each batch
    '''
    idx_list = np.arange(0, total_num, batch_size)  # strat of each batch_size
    if shuffle:   np.random.shuffle(idx_list)
    batches = []
    for idx in idx_list:
        batches.append(np.arange(idx, min(idx+batch_size, total_num)))
    return batches
    
def index_sentence(batch):
    '''
    Align sentence length
    :return x: sentences after aligning
    :return x_len: len of sentences
    '''
    batch_size = len(batch)
    sentences_len = [len(example) for example in batch]
    max_len = max(sentences_len)  # longest sentence in this batch

    x = np.ones((batch_size, max_len)).astype('int32')  # batch_size, max_len   PAD
    x_len = np.array(sentences_len).astype('int32')

    for index, sentence in enumerate(batch):
        x[index, :sentences_len[index]] = sentence
    
    return x, x_len

def gen_examples(en_sentences, cn_sentences, batch_size):
    '''
    :return batches: [batch*(data_len/batch_size)]  batch:(x,x_len, y,y_len)
    '''
    idx_batches = get_batches(len(en_sentences), batch_size)  # get the index of sentences for each batch
    batches = []
    for batch in idx_batches:
        b_en = [en_sentences[t] for t in batch] # index of sentence batch -> sentence batch
        b_cn = [cn_sentences[t] for t in batch]
        x, x_len = index_sentence(b_en)  # Align sentence length
        y, y_len = index_sentence(b_cn)
        batches.append((x, x_len, y, y_len))
    return batches

if __name__ == "__main__":
    # load data
    train_en, train_cn = load_data(train_file)
    dev_en, dev_cn = load_data(dev_file)
    print(dev_en[40], dev_cn[40])

    # build vocab
    en2idx, idx2en = build_vocab(train_en)
    cn2idx, idx2cn = build_vocab(train_cn)
    print('the index of "i" in en2idx is:', en2idx['i'])

    # # sentences to idxseqences
    train_en = sentences2idx(train_en, en2idx)
    train_cn = sentences2idx(train_cn, cn2idx)
    dev_en = sentences2idx(dev_en, en2idx)
    dev_cn = sentences2idx(dev_cn, cn2idx)
    print(train_en[40], train_cn[40])

    # get batch index
    # get_batches(100, 15)

    # batch_size = 64
    # train_data = gen_examples(train_en, train_cn, batch_size)
    # print(train_data)
    