#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import *

import numpy as np

import sys
sys.path.append('../tools')
import parse, py_op

def value_embedding_data(d = 512, split = 200):
    vec = np.array([np.arange(split) * i for i in range(d/2)], dtype=np.float32).transpose()
    vec = vec / vec.max() 
    embedding = np.concatenate((np.sin(vec), np.cos(vec)), 1)
    embedding[0, :d] = 0
    embedding = torch.from_numpy(embedding)
    return embedding

class AutoEncoder(nn.Module):
    def __init__(self, args):
        super(AutoEncoder, self).__init__()
        self.args = args

        if args.value_embedding == 'no':
            self.embedding = nn.Linear(args.output_size, args.embed_size)
        else:
            self.embedding = nn.Embedding (args.vocab_size, args.embed_size )
        self.lstm = nn.LSTM ( input_size=args.embed_size,
                              hidden_size=args.hidden_size,
                              num_layers=args.num_layers,
                              batch_first=True,
                              bidirectional=args.brnn)
        if args.dataset in ['MIMIC']:
            self.dd_embedding = nn.Embedding (args.n_ehr, args.embed_size )
        self.value_embedding = nn.Embedding.from_pretrained(value_embedding_data(args.embed_size, args.split_num + 1))
        self.value_mapping = nn.Sequential(
                nn.Linear ( args.embed_size * 2, args.embed_size),
                nn.ReLU ( ),
                nn.Dropout ( 0.1),
                )
        self.pre_embedding = nn.Sequential(
                nn.Linear ( args.embed_size * 2, args.embed_size),
                nn.ReLU ( ),
                nn.Linear ( args.embed_size, args.embed_size),
                nn.ReLU ( ),
                )
        self.post_embedding = nn.Sequential(
                nn.Linear ( args.embed_size * 2, args.embed_size),
                nn.ReLU ( ),
                nn.Linear ( args.embed_size, args.embed_size),
                nn.ReLU ( ),
                )
        self.pre_mapping = nn.Sequential(
                nn.Linear ( args.embed_size, args.embed_size),
                nn.ReLU ( ),
                nn.Linear ( args.embed_size, args.embed_size),
                nn.ReLU ( ),
                )
        self.post_mapping = nn.Sequential(
                nn.Linear ( args.embed_size, args.embed_size),
                nn.ReLU ( ),
                nn.Linear ( args.embed_size, args.embed_size),
                nn.ReLU ( ),
                )
        self.dd_mapping = nn.Sequential(
                nn.Linear ( args.embed_size, args.embed_size),
                nn.ReLU ( ),
                nn.Dropout(0.1),
                nn.Linear ( args.embed_size, args.embed_size),
                nn.ReLU ( ),
                nn.Dropout(0.1),
                )
        self.mapping = nn.Sequential(
                nn.Linear ( args.embed_size, args.embed_size),
                nn.ReLU ( ),
                nn.Linear ( args.embed_size, args.embed_size),
                nn.ReLU ( ),
                )
            # self.neib_embedding = nn.Sequential(
            #     nn.Linear ( args.embed_size * 2, args.embed_size),
            #     nn.ReLU ( ),
            #     nn.Dropout ( 0.1),
            #     )

        self.embed_linear = nn.Sequential (
            nn.Linear ( args.embed_size, args.embed_size),
            nn.ReLU ( ),
            # nn.Dropout ( 0.25 ),
            # nn.Linear ( args.embed_size, args.embed_size),
            # nn.ReLU ( ),
            nn.Dropout ( 0.1),
        )
        self.relu = nn.ReLU ( )

        lstm_size = args.rnn_size 
        if args.brnn:
            lstm_size *= 2
        hidden_size = args.hidden_size
        self.tah_mapping = nn.Sequential (
                nn.Linear(lstm_size, args.hidden_size),
                nn.ReLU ( ),
                nn.Dropout ( 0.1),
                nn.Linear ( args.hidden_size, hidden_size),
        ) 
        self.tav_mapping = nn.Sequential (
                nn.Linear(args.hidden_size, args.hidden_size),
                nn.ReLU ( ),
                nn.Dropout ( 0.1),
                nn.Linear ( args.hidden_size, hidden_size),
        ) 
        self.output = nn.Sequential (
            nn.Linear (lstm_size, args.rnn_size),
            nn.ReLU ( ),
            nn.Dropout ( 0.1),
            nn.Linear ( args.rnn_size, hidden_size),
        )
        self.value = nn.Sequential (
            nn.Linear (hidden_size, hidden_size),
            nn.ReLU ( ),
            nn.Dropout ( 0.1),
            nn.Linear (hidden_size, args.output_size),
        )
        self.pooling = nn.AdaptiveMaxPool1d(1)

    def visit_pooling(self, x):
        output = x
        size = output.size()
        output = output.view(size[0] * size[1], size[2], output.size(3))    # (64*30, 13, 512)
        output = torch.transpose(output, 1,2).contiguous()                  # (64*30, 512, 13)
        output = self.pooling(output)                                       # (64*30, 512, 1)
        output = output.view(size[0], size[1], size[3])                     # (64, 30, 512)
        return output

    def value_order_embedding(self, x):
        size = list(x[0].size())               # (64, 30, 13)
        index, value = x
        xi = self.embedding(index.view(-1))          # (64*30*13, 512)
        # xi = xi * (value.view(-1).float() + 1.0 / self.args.split_num)
        xv = self.value_embedding(value.view(-1))    # (64*30*13, 512)
        x = torch.cat((xi, xv), 1)                   # (64*30*13, 1024)
        x = self.value_mapping(x)                    # (64*30*13, 512)   
        size.append(-1)
        x = x.view(size)                    # (64, 30, 13, 512)
        return x

    def pp_value_embedding(self, neib):
        size = list(neib[1].size())
        # print(type(neib[0]))
        # print(len(neib[0]))
        if self.args.use_ve == 0:
            pre_x = self.embedding(neib[0])
            post_x = self.embedding(neib[2])
            pre_x = self.pre_mapping(pre_x)
            post_x = self.post_mapping(post_x)
        else:
            pre_x = self.value_order_embedding(neib[0])
            post_x = self.value_order_embedding(neib[2])

            pre_t = self.value_embedding(neib[1].view(-1))
            post_t = self.value_embedding(neib[3].view(-1))
            size.append(-1)
            pre_t = pre_t.view(size)
            post_t = post_t.view(size)

            pre_x = self.pre_embedding(torch.cat((pre_x, pre_t), 3))
            post_x = self.post_embedding(torch.cat((post_x, post_t), 3))
        return pre_x, post_x

    def time_aware_attention(self, hidden, vectors):
        # hidden [64, 30, 1024]
        # vectors [64, 30, 54, 512]
        wh = self.tah_mapping(hidden)
        wh = wh.unsqueeze(2)
        wh = wh.expand_as(vectors)
        wv = self.tav_mapping(vectors)
        beta = wh + wv # [64, 30, 54, 512]
        alpha = F.softmax(beta, 2)

        alpha = alpha.transpose(2,3).contiguous()
        vectors = vectors.transpose(2,3).contiguous()
        vsize = list(vectors.size()) # [64, 30, 512, 54]
        # print(alpha.size())
        # print(vectors.size())
        alpha = alpha.view((-1, 1, alpha.size(3)))
        vectors = vectors.view((-1, vectors.size(3), 1))
        # print(alpha.size())
        # print(vectors.size())
        att_res = torch.bmm(alpha, vectors)
        # print(att_res.size())
        att_res = att_res.view(vsize[:3])
        # print(att_res.size())
        # return att_res
        return att_res



    def forward(self, x, neib=None, dd=None, mask=None):
        # print()

        # embedding
        if self.args.value_embedding == 'no':
            x = self.embedding( x )             # (64, 30, 512)
        else:
            if self.args.value_embedding == 'use_order':
                x = self.value_order_embedding(x)
                    # x = self.neib_embedding(torch.cat((x, pre_x, post_x), 3))
            else:
                size = list(x.size())               # (64, 30, 13)
                x = x.view(-1)
                x = self.embedding( x )             # (64*30*13, 512)


        # print(x.size())

        if dd is not None:
            dsize = list(dd.size()) + [-1]
            d = self.dd_embedding(dd.view(-1)).view(dsize)
            d = self.dd_mapping(d)
            x = torch.cat((x, d), 2)
            x = self.mapping(x)

        x = self.visit_pooling(x)           # (64, 30, 512)

        # lstm
        lstm_out, _ = self.lstm( x )            # (64, 30, 1024)
        out = self.output(lstm_out)

        if neib is not None and self.args.use_ta:
            pre_x, post_x = self.pp_value_embedding(neib)
            pp = torch.cat((pre_x, post_x), 2)
            out = out + self.time_aware_attention(lstm_out, pp)
        else:
            pre_x, post_x = self.pp_value_embedding(neib)
            pp = torch.cat((pre_x, post_x), 2)
            pp = self.visit_pooling(pp)
            out = out + pp

        value = self.value(out)

        return value

