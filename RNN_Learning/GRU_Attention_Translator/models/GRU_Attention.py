#! -*- encoding:utf-8 -*-
"""
@File    :   GRU_Attention.py
@Author  :   Zachary Li
@Contact :   li_zaaachary@163.com
@Dscpt   :   
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        
    def forward(self, x, x_lengths, y, y_lengths):
        encoder_out, hid = self.encoder(x, x_lengths)
        # print(hid.shape)=torch.Size([1, batch_size, dec_hidden_size])
        # print(out.shape)=torch.Size([batch_size, seq_len, 2*enc_hidden_size])
        output, hid, attn = self.decoder(encoder_out=encoder_out, 
                    x_lengths=x_lengths,
                    y=y,
                    y_lengths=y_lengths,
                    hid=hid)
        # output =(batch_size, output_len, vocab_size)
        # hid.shape = (1, batch_size, dec_hidden_size)
        # attn.shape = (batch_size, output_len, context_len)
        return output, attn
    

    def translate(self, x, x_lengths, y, max_length=100):
        encoder_out, hid = self.encoder(x, x_lengths)
        preds = []
        batch_size = x.shape[0]
        attns = []
        for i in range(max_length):
            output, hid, attn = self.decoder(encoder_out, 
                    x_lengths,
                    y,
                    torch.ones(batch_size).long().to(y.device),
                    hid)
            y = output.max(2)[1].view(batch_size, 1)
            preds.append(y)
            attns.append(attn)
            
        return torch.cat(preds, 1), torch.cat(attns, 1)


class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_size, enc_hidden_size, dec_hidden_size, dropout=0.2):
        super(Encoder, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size, enc_hidden_size, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(enc_hidden_size * 2, dec_hidden_size)

    def forward(self, x, lengths):
        sorted_len, sorted_idx = lengths.sort(0, descending=True)
        x_sorted = x[sorted_idx.long()]
        embedded = self.dropout(self.embed(x_sorted))
        
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, sorted_len.long().cpu().data.numpy(), batch_first=True)
        packed_out, hid = self.rnn(packed_embedded)
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        _, original_idx = sorted_idx.sort(0, descending=False)
        out = out[original_idx.long()].contiguous()
        hid = hid[:, original_idx.long()].contiguous()
        # hid: [2, batch_size, enc_hidden_size]
        
        hid = torch.cat([hid[-2], hid[-1]], dim=1) # 将最后一层的hid的双向拼接
        # hid: [batch_size, 2*enc_hidden_size]
        hid = torch.tanh(self.fc(hid)).unsqueeze(0)
        # hid: [1, batch_size, dec_hidden_size]
        # out: [batch_size, seq_len, 2*enc_hidden_size]
        return out, hid


class Attention(nn.Module):
    def __init__(self, enc_hidden_size, dec_hidden_size):
        # enc_hidden_size跟Encoder的一样
        super(Attention, self).__init__()
        self.enc_hidden_size = enc_hidden_size
        self.dec_hidden_size = dec_hidden_size

        self.linear_in = nn.Linear(enc_hidden_size*2, dec_hidden_size, bias=False)
        self.linear_out = nn.Linear(enc_hidden_size*2 + dec_hidden_size, dec_hidden_size)
        
    def forward(self, output, context, mask):
        # mask = batch_size, output_len, context_len     # mask在Decoder中创建好了
        # output: batch_size, output_len, dec_hidden_size，就是Decoder的output
        # context: batch_size, context_len, 2*enc_hidden_size，就是Encoder的output 
        # 这里Encoder网络是双向的，Decoder是单向的
    
        batch_size = output.size(0)
        output_len = output.size(1)
        input_len = context.size(1) # input_len = context_len
        
        # 通过decoder的hidden states加上encoder的hidden states来计算一个分数，用于计算权重
        # batch_size, context_len, dec_hidden_size
        # 第一步，公式里的Wa先与hs做点乘，把Encoder output的enc_hidden_size换成dec_hidden_size。
        # Q: W·context
        context_in = self.linear_in(context.view(batch_size*input_len, -1)).view(                
                                    batch_size, input_len, -1) 
        
        # Q·K
        # context_in.transpose(1,2): batch_size, dec_hidden_size, context_len 
        # output: batch_size, output_len, dec_hidden_size
        attn = torch.bmm(output, context_in.transpose(1,2)) 
        # batch_size, output_len, context_len
        # 第二步，ht与上一步结果点乘，得到score

        attn.data.masked_fill(mask, -1e6)
        # .masked_fill作用请看这个链接：https://blog.csdn.net/candy134834/article/details/84594754
        # mask的维度必须和attn维度相同，mask为1的位置对应attn的位置的值替换成-1e6，
        # mask为1的意义需要看Decoder函数里面的定义

        attn = F.softmax(attn, dim=2) 
        # batch_size, output_len, context_len
        # 这个dim=2到底是怎么softmax的看下下面单元格例子
        # 第三步，计算每一个encoder的hidden states对应的权重。
        
        # context: batch_size, context_len, 2*enc_hidden_size，
        context = torch.bmm(attn, context) 
        # batch_size, output_len, 2*enc_hidden_size
        # 第四步，得出context vector是一个对于encoder输出的hidden states的一个加权平均
        
        # output: batch_size, output_len, dec_hidden_size
        output = torch.cat((context, output), dim=2) 
        # output：batch_size, output_len, 2*enc_hidden_size+dec_hidden_size
        # 第五步，将context vector和 decoder的hidden states 串起来。
        
        output = output.view(batch_size*output_len, -1)
        # output.shape = (batch_size*output_len, 2*enc_hidden_size+dec_hidden_size)
        output = torch.tanh(self.linear_out(output)) 
        # output.shape=(batch_size*output_len, dec_hidden_size)
        output = output.view(batch_size, output_len, -1)
        # output.shape=(batch_size, output_len, dec_hidden_size)
        # attn.shape = batch_size, output_len, context_len
        return output, attn


class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_size, enc_hidden_size, dec_hidden_size, dropout=0.2):
        super(Decoder, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.attention = Attention(enc_hidden_size, dec_hidden_size)
        self.rnn = nn.GRU(embed_size, enc_hidden_size, batch_first=True)
        self.out = nn.Linear(dec_hidden_size, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def create_mask(self, x_len, y_len):
        # x_len 是一个batch中文句子的长度列表
        # y_len 是一个batch英文句子的长度列表
        # a mask of shape x_len * y_len
        device = x_len.device
        max_x_len = x_len.max()
        max_y_len = y_len.max()
        
        x_mask = torch.arange(max_x_len, device=device)[None, :] < x_len[:, None]
        # print(x_mask.shape) = (batch_size, output_len) # 中文句子的mask
        y_mask = torch.arange(max_y_len, device=device)[None, :] < y_len[:, None]
        # print(y_mask.shape) = (batch_size, context_len) # 英文句子的mask
        
        mask = ( ~ x_mask[:, :, None] * y_mask[:, None, :]).bool()
        # mask = (1 - x_mask[:, :, None] * y_mask[:, None, :]).byte()
        # 1-说明取反
        # x_mask[:, :, None] = (batch_size, output_len, 1)
        # y_mask[:, None, :] =  (batch_size, 1, context_len)
        # print(mask.shape) = (batch_size, output_len, context_len)
        # 注意这个例子的*相乘不是torch.bmm矩阵点乘，只是用到了广播机制而已。
        return mask
    
    def forward(self, encoder_out, x_lengths, y, y_lengths, hid):
        sorted_len, sorted_idx = y_lengths.sort(0, descending=True)
        y_sorted = y[sorted_idx.long()]
        hid = hid[:, sorted_idx.long()]
        
        y_sorted = self.dropout(self.embed(y_sorted)) # batch_size, output_length, embed_size

        packed_seq = nn.utils.rnn.pack_padded_sequence(y_sorted, sorted_len.long().cpu().data.numpy(), batch_first=True)
        out, hid = self.rnn(packed_seq, hid)
        unpacked, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        _, original_idx = sorted_idx.sort(0, descending=False)
        output_seq = unpacked[original_idx.long()].contiguous()
        hid = hid[:, original_idx.long()].contiguous()

        mask = self.create_mask(y_lengths, x_lengths) # 这里真是坑，第一个参数位置是中文句子的长度列表

        output, attn = self.attention(output_seq, encoder_out, mask) 
        # output.shape=(batch_size, output_len, dec_hidden_size)
        # attn.shape = batch_size, output_len, context_len
        
        # self.out = nn.Linear(dec_hidden_size, vocab_size)
        output = F.log_softmax(self.out(output), -1) # 计算最后的输出概率
        # output =(batch_size, output_len, vocab_size)
        # 最后一个vocab_size维度 log_softmax
        # hid.shape = (1, batch_size, dec_hidden_size)
        return output, hid, attn


def make_model(en_len, cn_len):
    dropout = 0.2
    embed_size = hidden_size = 100
    encoder = Encoder(vocab_size=en_len,
                        embed_size=embed_size,
                        enc_hidden_size=hidden_size,
                        dec_hidden_size=hidden_size,
                        dropout=dropout)
    decoder = Decoder(vocab_size=cn_len,
                        embed_size=embed_size,
                        enc_hidden_size=hidden_size,
                        dec_hidden_size=hidden_size,
                        dropout=dropout)
    model = Seq2Seq(encoder, decoder)
    return model