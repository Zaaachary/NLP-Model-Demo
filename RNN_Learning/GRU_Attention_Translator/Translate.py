#! -*- encoding:utf-8 -*-
"""
@File    :   Translate.py
@Author  :   Zachary Li
@Contact :   li_zaaachary@163.com
@Dscpt   :   
"""
import random

import torch

from models import Plain_RNN, Criterion
from PrepareData import *

def translate_dev(model, dev_en):
    # 一条句子
    mb_x = torch.from_numpy(np.array(dev_en).reshape(1, -1)).long().to(device)
    mb_x_len = torch.from_numpy(np.array([len(dev_en)])).long().to(device)
    bos = torch.Tensor([[cn2idx["BOS"]]]).long().to(device)  # shape:[1,1], [[2]]
    
    # y_lengths: [[2]], 一个句子
    translation, attn = model.translate(mb_x, mb_x_len, bos)  # [1, 10]
    # 映射成中文
    # print(translation)
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
    # load from pickle
    data = autoload()
    en2idx = data['en2idx']
    cn2idx = data['cn2idx']
    idx2en = data['idx2en']
    idx2cn = data['idx2cn']
    dev_en = data['dev_en']
    dev_cn = data['dev_cn']

    # ===== DEFINE MODEL ===== #
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dropout, hidden_size = 0.05, 100
    # encoder = Plain_RNN.PlainEncoder(len(en2idx), hidden_size, dropout)
    # decoder = Plain_RNN.PlainDecoder(len(en2idx), hidden_size, dropout)
    encoder = Plain_RNN.PlainEncoder(len(en2idx), hidden_size, 2, dropout)
    decoder = Plain_RNN.PlainDecoder(len(en2idx), hidden_size, 2, dropout)
    model = Plain_RNN.PlainSeq2Seq(encoder, decoder)

    model.to(device)
    model.eval() # 关闭dropout
    # 导入训练好模型
    model.load_state_dict(torch.load('./checkpoint/Plain_RNN2.pt', map_location=device))
    for i in range(10):
        # i = random.choice()
        en_sent = " ".join([idx2en[w] for w in dev_en[i]][1:-1])  #原来的英文
        print('{: <20}:'.format('original sentence'), en_sent)
        cn_sent = " ".join([idx2cn[w] for w in dev_cn[i]][1:-1])  #原来的中文
        print('{: <20}:'.format('true translation'), cn_sent)
        translate_dev(model, dev_en[i])
        print()
