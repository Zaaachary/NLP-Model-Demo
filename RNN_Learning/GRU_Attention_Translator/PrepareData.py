#! -*- encoding:utf-8 -*-
"""
@File    :   PrepareData.py
@Author  :   Zachary Li
@Contact :   li_zaaachary@163.com
@Dscpt   :   Reference: https://www.cnblogs.com/douzujun/p/13624567.html
"""

import os
import sys
import math
import random
import pickle
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import nltk

# nltk.download('punkt')

train_file = './data_spcn/train.txt'
dev_file = './data_spcn/dev.txt'
pickle_file = './data_spcn/data.pk'


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

def sentences2idx(en_sentences, cn_sentences, en_word2idx, cn_word2idx, len_sort=False):
    '''
    transform word sentence 2 idx sentence, and sort by en
    '''
    length = len(en_sentences)
    # word sentence 2 idx sentence
    en_idx_sentences = [[en_word2idx.get(word, 0) for word in sentence] for sentence in en_sentences]
    cn_idx_sentences = [[cn_word2idx.get(word, 0) for word in sentence] for sentence in cn_sentences]
    # sort by len
    if len_sort:
        sorted_index = sorted(range(length), 
            key=lambda index: len(en_idx_sentences[index])+len(cn_idx_sentences[index])
            )
        en_idx_sentences = [en_idx_sentences[index] for index in sorted_index]
        cn_idx_sentences = [cn_idx_sentences[index] for index in sorted_index]
    return en_idx_sentences, cn_idx_sentences

def get_batches(total_num, batch_size, shuffle=True):
    '''
    shuffle and return index of sentences for each batch
    e.g.  10, 3 -> [4,5,6],[1,2,3],[7,8,9],[10]
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
    sentences_len = [len(example) for example in batch] # length of each sentence
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
        b_en = [en_sentences[t] for t in batch]  # index of sentence batch -> words of sentence batch
        b_cn = [cn_sentences[t] for t in batch]
        x, x_len = index_sentence(b_en)  # Align sentence length
        y, y_len = index_sentence(b_cn)
        batches.append((x, x_len, y, y_len))
    return batches

def autoload():
    '''
    en2idx, cn2idx, idx2en, idx2cn,
    '''
    if os.path.exists(pickle_file):
        with open(pickle_file, 'rb') as f:
            data = pickle.load(f)
    else:
        data = generate()
        with open(pickle_file, 'wb') as f:
            pickle.dump(data, f)
    return data

def generate():
    data = {}
    # load data
    train_en_w, train_cn_w = load_data(train_file)  # _w means word
    dev_en_w, dev_cn_w = load_data(dev_file)
    print(dev_en_w[40])
    print(dev_cn_w[40])

    # build vocab
    en2idx, idx2en = build_vocab(train_en_w)
    cn2idx, idx2cn = build_vocab(train_cn_w)
    data['en2idx'] = en2idx
    data['cn2idx'] = cn2idx
    data['idx2en'] = idx2en
    data['idx2cn'] = idx2cn
    print('the index of "i" in en2idx is:', en2idx['i'])

    # sentences to idxseqences
    train_en, train_cn = sentences2idx(train_en_w, train_cn_w, en2idx, cn2idx, len_sort=False)
    dev_en, dev_cn = sentences2idx(dev_en_w, dev_cn_w, en2idx, cn2idx, len_sort=False)
    data['train_en'] = train_en
    data['train_cn'] = train_cn
    data['dev_en'] = dev_en
    data['dev_cn'] = dev_cn
    print('words:', train_en_w[30], 'index:', train_en[30])
    print('words:', train_cn_w[30], 'index:', train_cn[30])

    # test get batch
    # get_batches(100, 15)

    batch_size = 128
    train_data = gen_examples(train_en, train_cn, batch_size)
    dev_data = gen_examples(dev_en, dev_cn, batch_size)
    data['train_data'] = train_data
    data['dev_data'] = dev_data
    print('num of batchs:', len(train_data))
    print('maxlen of No.200 batch:', train_data[-1][0][0].size)
    
    return data


if __name__ == "__main__":
    data = autoload()
    print(data.keys())
    
    # unpack
    en2idx = data['en2idx']
    cn2idx = data['cn2idx']
    idx2en = data['idx2en']
    idx2cn = data['idx2cn']
    train_en = data['train_en']
    train_cn = data['train_cn']
    dev_en = data['dev_en']
    dev_cn = data['dev_cn']
    train_data = data['train_data']
    dev_data = data['dev_data']
