#!/usr/bin/env python
# coding=utf-8


import sys
reload(sys)
sys.setdefaultencoding('utf8')

import os
import sys
import time
import numpy as np
from sklearn import metrics
import random
import json
from glob import glob
from collections import OrderedDict
from tqdm import tqdm


sys.path.append('../tools')
import parse, py_op
args = parse.args

def generate_feature_mm_dict():
    files = sorted(glob(os.path.join(args.data_dir, args.dataset, 'train_groundtruth/*')))
    feature_value_dict = dict()
    for ifi, fi in enumerate(tqdm(files)):
        if 'csv' not in fi:
            continue
        for iline, line in enumerate(open(fi)):
            line = line.strip()
            if iline == 0:
                feat_list = line.split(',')
            else:
                data = line.split(',')
                for iv, v in enumerate(data):
                    if v in ['NA', '']:
                        continue
                    else:
                        feat = feat_list[iv]
                        if feat not in feature_value_dict:
                            feature_value_dict[feat] = []
                        feature_value_dict[feat].append(float(v))
    feature_mm_dict = dict()
    feature_ms_dict = dict()

    feature_range_dict = dict()
    for feat, vs in feature_value_dict.items():
        vs = sorted(vs)
        value_split = []
        for i in range(args.split_num):
            n = int(i * len(vs) / args.split_num)
            value_split.append(vs[n])
        value_split.append(vs[-1])
        feature_range_dict[feat] = value_split


        n = int(len(vs) / args.split_num)
        feature_mm_dict[feat] = [vs[n], vs[-n - 1]]
        feature_ms_dict[feat] = [np.mean(vs), np.std(vs)]

    py_op.mkdir(args.file_dir)
    py_op.mywritejson(os.path.join(args.file_dir, args.dataset + '_feature_mm_dict.json'), feature_mm_dict)
    py_op.mywritejson(os.path.join(args.file_dir, args.dataset + '_feature_ms_dict.json'), feature_ms_dict)
    py_op.mywritejson(os.path.join(args.file_dir, args.dataset + '_feature_list.json'), feat_list)
    py_op.mywritejson(os.path.join(args.file_dir, args.dataset + '_feature_value_dict_{:d}.json'.format(args.split_num)), feature_range_dict)

def split_data_to_ten_set():
    files = sorted(glob(os.path.join(args.data_dir, args.dataset, 'train_with_missing/*')))
    np.random.shuffle(files)
    splits = []
    for i in range(10):
        st = int(len(files) * i / 10)
        en = int(len(files) * (i+1) / 10)
        splits.append(files[st:en])
    py_op.mywritejson(os.path.join(args.file_dir, args.dataset + '_splits.json'), splits)


def main():
    generate_feature_mm_dict()
    split_data_to_ten_set()

if __name__ == '__main__':
    main()
