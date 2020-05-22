# coding=utf8

import os
import argparse

parser = argparse.ArgumentParser(description='MIMIC III PROJECTS')

# data dir
parser.add_argument(
        '--data-dir',
        type=str,
        default='../../data/',
        help='selected and preprocessed data directory'
        )
parser.add_argument(
        '--result-dir',
        type=str,
        default='../../result/',
        help='result directory'
        )
parser.add_argument(
        '--file-dir',
        type=str,
        default='../../file/',
        help='useful file directory'
        )
parser.add_argument(
        '--mimic-dir',
        type=str,
        default='../../data/MIMIC/initial_mimiciii/',
        help='useful file directory'
        )

parser.add_argument(
        '--dataset',
        default='DACMI',
        # default='MIMIC',
        type=str,
        help='dataset')

parser.add_argument(
        '--n-code',
        default=8,
        type=int,
        help='at most n codes for same visit')
parser.add_argument(
        '--n-visit',
        default=30,
        type=int,
        help='at most input n visits')
parser.add_argument(
        '--nc',
        default=4,
        type=int,
        help='n clusters')
parser.add_argument(
        '--brnn',
        default=True,
        type=bool,
        help='use bidirectional RNN or not')
parser.add_argument(
        '--random-missing',
        default=True,
        type=bool,
        help='use random missing values for training')



# method seetings
parser.add_argument(
        '--model',
        '-m',
        type=str,
        default='tame',
        help='model'
        )
parser.add_argument(
        '--split-num',
        metavar='split num',
        type=int,
        default=4000,
        help='split num'
        )
parser.add_argument(
        '--n-records',
        metavar='input size',
        type=int,
        default=30,
        help='input size'
        )
parser.add_argument(
        '--split-nor',
        metavar='split normal range',
        type=int,
        default=3,
        help='split num'
        )
parser.add_argument(
        '--use-ta',
        metavar='use time-aware attention',
        type=int,
        default=1,
        help='use time-aware attention'
        )
parser.add_argument(
        '--use-ve',
        metavar='use value embedding',
        type=int,
        default=1,
        help='use value-embedding'
        )
parser.add_argument(
        '--use-mm',
        metavar='use multi-modal input',
        type=int,
        default=0,
        help='use multi-modal input'
        )
parser.add_argument(
        '--value-embedding',
        metavar='use time embedding',
        type=str,
        # default='use_value',
        default='use_order',
        # default='no',
        help='use_value or use_order or no'
        )
parser.add_argument(
        '--loss',
        type=str,
        # default='missing',
        # default='init',
        default='both',
        help='loss function, missing, init, both'
        )


# model parameters
parser.add_argument(
        '--embed-size',
        metavar='EMBED SIZE',
        type=int,
        default=512,
        help='embed size'
        )
parser.add_argument(
        '--rnn-size',
        metavar='rnn SIZE',
        type=int,
        help='rnn size'
        )
parser.add_argument(
        '--hidden-size',
        metavar='hidden SIZE',
        type=int,
        help='hidden size'
        )
parser.add_argument(
        '--num-layers',
        metavar='num layers',
        type=int,
        default=2,
        help='num layers'
        )



# traing process setting
parser.add_argument('--phase',
        default='train',
        type=str,
        metavar='S',
        help='pretrain/train/test phase')
parser.add_argument(
        '--batch-size',
        '-b',
        metavar='BATCH SIZE',
        type=int,
        default=64,
        help='batch size'
        )
parser.add_argument('--resume',
        default='',
        type=str,
        metavar='S',
        help='start from checkpoints')
parser.add_argument(
        '--compute-weight',
        default=0,
        type=int,
        help='compute weight for interpretebility')
parser.add_argument(
        '--workers',
        default=16,
        type=int,
        metavar='N',
        help='number of data loading workers (default: 32)')
parser.add_argument('--lr',
        '--learning-rate',
        default=0.001,
        type=float,
        metavar='LR',
        help='initial learning rate')
parser.add_argument('--epochs',
        default=2000,
        type=int,
        metavar='N',
        help='number of total epochs to run')
parser.add_argument('--save-freq',
        default=1,
        type=int,
        metavar='S',
        help='save frequency')
parser.add_argument('--save-pred-freq',
        default='10',
        type=int,
        metavar='S',
        help='save pred clean frequency')
parser.add_argument('--val-freq',
        default=1,
        type=int,
        metavar='S',
        help='val frequency')

args = parser.parse_args()
