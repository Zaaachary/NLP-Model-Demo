#! -*- encoding:utf-8 -*-
"""
@File    :   Translate.py
@Author  :   Zachary Li
@Contact :   li_zaaachary@163.com
@Dscpt   :   
"""

import torch

from models import Plain_RNN, Criterion
from PrepareData import *

def translate_dev(i):
    en_sent = " ".join([idx2en[w] for w in dev_en[i]][1:-1])  #原来的英文
    print('{: <20}:'.format('original sentence'), en_sent)
    cn_sent = " ".join([idx2cn[w] for w in dev_cn[i]][1:-1])  #原来的中文
    print('{: <20}:'.format('true translation'), cn_sent)

    # 一条句子
    mb_x = torch.from_numpy(np.array(dev_en[i]).reshape(1, -1)).long().to(device)
    mb_x_len = torch.from_numpy(np.array([len(dev_en[i])])).long().to(device)
    bos = torch.Tensor([[cn2idx["BOS"]]]).long().to(device)  # shape:[1,1], [[2]]
    
    # y_lengths: [[2]], 一个句子
    translation, attn = model.translate(mb_x, mb_x_len, bos,)  # [1, 10]
    # 映射成中文
    translation = [idx2cn[i] for i in translation.data.cpu().numpy().reshape(-1)]
    trans = []
    for word in translation:
        if word != "EOS":
            trans.append(word)
        else:
            break
    print('{: <20}:'.format('model prediction')," ".join(trans))           #翻译后的中文


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
    batch_size = 64
    train_data = gen_examples(train_en, train_cn, batch_size)
    dev_data = gen_examples(dev_en, dev_cn, batch_size)

    # ===== TRAIN MODEL ===== #
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dropout, hidden_size = 0.2, 100
    # define model
    encoder = Plain_RNN.PlainEncoder(len(en2idx), hidden_size, dropout)
    decoder = Plain_RNN.PlainDecoder(len(en2idx), hidden_size, dropout)
    model = Plain_RNN.PlainSeq2Seq(encoder, decoder)

    # 导入训练好模型
    model.load_state_dict(torch.load('./checkpoint/Plain_RNN.pt', map_location=device))
    for i in range(100, 120):
        translate_dev(i)
        print()
