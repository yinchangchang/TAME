#!/usr/bin/env python
# -*- coding: utf-8 -*-
#

import json
import collections
import os
import random
import time
import warnings
from copy import deepcopy

import numpy as np
import torch
from torch.utils.data import Dataset

import sys
sys.path.append('../tools')
import py_op

def find_index(v, vs, i=0, j=-1):
    if j == -1:
        j = len(vs) - 1

    if v > vs[j]:
        return j + 1
    elif v < vs[i]:
        return i
    elif j - i == 1:
        return j

    k = int((i + j)/2)
    if v <= vs[k]:
        return find_index(v, vs, i, k)
    else:
        return find_index(v, vs, k, j)

def add_time_gap(idata, odata, n = 10):
    '''
    delete lines  with only meanbp_min, and urineoutput
    '''
    new_idata = []
    new_odata = []
    for iline,oline in zip(idata, odata):
        vs = []
        for v in oline.strip().split(','):
            if v in ['', 'NA']:
                vs.append(0)
            else:
                vs.append(1)
        vs[0] = 0
        vs[6] = 0
        vs[8] = 0
        if np.sum(vs) > 0:
            new_idata.append(iline)
            new_odata.append(oline)
    return new_idata, new_odata


class DataBowl(Dataset):
    def __init__(self, args, files, phase='train'):
        assert (phase == 'train' or phase == 'valid' or phase == 'test')
        self.args = args
        self.phase = phase
        self.files = files

        self.feature_mm_dict = py_op.myreadjson(os.path.join(args.file_dir, args.dataset + '_feature_mm_dict.json'))
        self.feature_value_dict = py_op.myreadjson(os.path.join(args.file_dir, args.dataset + '_feature_value_dict_{:d}.json'.format(args.split_num)))
        self.n_dd = 40
        if args.dataset in ['MIMIC']:
            self.ehr_list = py_op.myreadjson(os.path.join(args.data_dir, args.dataset, 'ehr_list.json' ))
            self.ehr_id = { ehr: i+1 for i,ehr in enumerate(self.ehr_list) }
            self.args.n_ehr = len(self.ehr_id) + 1

    def map_input(self, value, feat_list, feat_index):

        # for each feature (index), there are 1 embedding vectors for NA, split_num=100 embedding vectors for different values
        index_start = (feat_index + 1)* (1 + self.args.split_num) + 1

        if value in ['NA', '']:
            if self.args.value_embedding == 'no':
                return 0
            return index_start - 1
        else:
            # print('""' + value + '""')
            value = float(value)
            if self.args.value_embedding == 'use_value':
                minv, maxv = self.feature_mm_dict[feat_list[feat_index]]
                v = (value - minv) / (maxv - minv + 10e-10)
                # print(v, minv, maxv)
                assert v >= 0
                # map the value to its embedding index
                v = int(self.args.split_num * v) + index_start
                return v
            elif self.args.value_embedding == 'use_order':
                vs = self.feature_value_dict[feat_list[feat_index]][1:-1]
                v = find_index(value, vs) + index_start
                return v
            elif self.args.value_embedding == 'no':
                minv, maxv = self.feature_mm_dict[feat_list[feat_index]]
                # v = (value - minv) / (maxv - minv)
                v = (value - minv) / maxv + 1
                v = int(v * self.args.split_num) / float(self.args.split_num)
                return v

    def map_output(self, value, feat_list, feat_index):
        if value in ['NA', '']:
            return 0
        else:
            value = float(value)
            minv, maxv = self.feature_mm_dict[feat_list[feat_index]]
            if maxv <= minv:
                print(feat_list[feat_index], minv, maxv)
            assert maxv > minv
            v = (value - minv) / (maxv - minv)
            # v = (value - minv) / (maxv - minv)
            v = max(0, min(v, 1))
            return v

    def get_pre_info(self, input_data, iline, feat_list):
        iline = len(input_data) - iline - 1
        input_data = input_data[::-1][: -1] # the first line is head
        pre_input, pre_time = self.get_post_info(input_data, iline, feat_list)
        return pre_input, pre_time
        

    def get_post_info(self, input_data, iline, feat_list):
        input_data = input_data[iline:]
        # input_data = [s.split(',') for s in input_data]
        post_input = [0]
        post_time = [0]
        for i in range(1, len(input_data[0])):
            value = ''
            t = 0
            for j in range(1, len(input_data)):
                if input_data[j][i] not in ['NA', '']:
                    value = input_data[j][i]
                    t = abs(int(input_data[j][0]) - int(input_data[0][0]))
                    break
            post_input.append(self.map_input(value, feat_list, i))
            post_time.append(t)
        return post_input, post_time



    def get_mm_item(self, idx):
        input_file = self.files[idx]
        output_file = input_file.replace('with_missing', 'groundtruth')

 


        with open(output_file) as f:
            output_data = f.read().strip().split('\n')
        with open(input_file) as f:
            input_data = f.read().strip().split('\n')





        if self.args.random_missing and self.phase == 'train':
            input_data = []
            valid_list = []
            for line in output_data:
                values = line.strip().split(',')
                input_data.append(values)
                valid = []
                for iv,v in enumerate(values):
                    if v.strip() not in ['', 'NA']:
                        valid.append(1)
                    else:
                        valid.append(0)
                valid_list.append(valid)
            valid_list = np.array(valid_list)
            valid_list[0] = 0
            for i in range(1, valid_list.shape[1]):
                valid = valid_list[:, i]
                indices = np.where(valid > 0)[0]
                np.random.shuffle(indices)
                if len(indices>2):
                    input_data[indices[0]][i] = 'NA'
            input_data = [','.join(line) for line in input_data]



        init_input_data = input_data

        if self.args.use_ve == 0:
            input_data = self.pre_filling(input_data)


        assert len(input_data) == len(output_data)

        mask_list, input_list, output_list = [], [], []
        pre_input_list, pre_time_list = [], []
        post_input_list, post_time_list = [], []
        input_split = [x.strip().split(',') for x in init_input_data]

        for iline in range(len(input_data)):
            inp = input_data[iline].strip()
            oup = output_data[iline].strip()
            init_inp = init_input_data[iline].strip()

            if iline == 0:
                feat_list = inp.split(',')
            else:
                in_vs = inp.split(',')
                ou_vs = oup.split(',')
                init_vs = init_inp.split(',')
                ctime = int(inp.split(',')[0])
                mask, input, output = [], [], []
                rd = np.random.random(len(in_vs))
                for i, (iv, ov, ir, init_iv) in enumerate(zip(in_vs, ou_vs, rd, init_vs)):
                    if ir < 0.7:
                        # iv = 'NA'
                        pass


                    if init_iv not in ['NA', '']:
                        mask.append(0)
                    elif ov  not in ['NA', '']:
                        # print('err')
                        mask.append(1)
                    else:
                        mask.append(-1)
                    if self.args.use_ve:
                        input.append(self.map_input(iv, feat_list, i))
                    else:
                        input.append(self.map_output(iv, feat_list, i))
                    output.append(self.map_output(ov, feat_list, i))
                mask_list.append(mask)
                input_list.append(input)
                output_list.append(output)
                # pre and post info
                pre_input, pre_time = self.get_pre_info(input_split, iline, feat_list)
                pre_input_list.append(pre_input)
                pre_time_list.append(pre_time)
                post_input, post_time = self.get_post_info(input_split, iline, feat_list)
                post_input_list.append(post_input)
                post_time_list.append(post_time)

        if len(mask_list) < self.args.n_visit:
            for _ in range(self.args.n_visit - len(mask_list)):
                # pad empty visit
                mask = [-1 for _ in range(self.args.output_size + 1)]
                vs = [0 for _ in range(self.args.output_size + 1)]
                mask_list.append(mask)
                input_list.append(vs)
                output_list.append(vs)
                pre_input_list.append(vs)
                pre_time_list.append(vs)
                post_input_list.append(vs)
                post_time_list.append(vs)
            # print(mask_list)
        else:
            mask_list = mask_list[: self.args.n_visit]
            input_list = input_list[: self.args.n_visit]
            output_list = output_list[: self.args.n_visit]
            pre_input_list = pre_input_list[: self.args.n_visit]
            pre_time_list = pre_time_list[: self.args.n_visit]
            post_input_list = post_input_list[: self.args.n_visit]
            post_time_list = post_time_list[: self.args.n_visit]







        # print(mask_list)
        mask_list = np.array(mask_list, dtype=np.int64)
        output_list = np.array(output_list, dtype=np.float32)
        pre_time_list = np.array(pre_time_list, dtype=np.int64)
        post_time_list = np.array(post_time_list, dtype=np.int64)
        if self.args.value_embedding == 'no' or self.args.use_ve == 0:
            input_list = np.array(input_list, dtype=np.float32)
            pre_input_list = np.array(pre_input_list, dtype=np.float32)
            post_input_list = np.array(post_input_list, dtype=np.float32)
        else:
            input_list = np.array(input_list, dtype=np.int64)
            pre_input_list = np.array(pre_input_list, dtype=np.int64)
            post_input_list = np.array(post_input_list, dtype=np.int64)

        input_list = input_list[:, 1:]
        output_list = output_list[:, 1:]
        mask_list = mask_list[:, 1:]
        pre_input_list = pre_input_list[:, 1:]
        pre_time_list = pre_time_list[:, 1:]
        post_input_list = post_input_list[:, 1:]
        post_time_list = post_time_list[:, 1:]

        time_list = [x[0] for x in input_split][1:]
        max_time = int(time_list[min(self.args.n_visit, len(time_list) - 1)]) + 1

        if self.args.dataset in ['MIMIC']:
            ehr_dict = py_op.myreadjson(os.path.join(input_file.replace('with_missing', 'groundtruth').replace('.csv', '.json')))
        else:
            ehr_dict = dict()
        icd_list = [self.ehr_id[e] for e in ehr_dict.get('icd_demo', { }) if e in self.ehr_id]
        icd_list = set(icd_list)
        icd_list = set()
        drug_dict = ehr_dict.get('drug', { })
        visit_dict = dict()
        for i in range(- 250, max_time + 1):
            visit_dict[i] = sorted(icd_list)
        for k, drug_list in drug_dict.items():
            stime, etime = k.split(' -- ')
            id_list = list(set([self.ehr_id[e] for e in drug_list if e in self.ehr_id]))
            if len(id_list):
                for t in range(max(0, int(stime)), int(etime)):
                    if t in visit_dict:
                        visit_dict[t] = visit_dict[t] + id_list
        for k,v in visit_dict.items():
            v = list(set(v))
            visit_dict[k] = v
            # if self.n_dd < len(v):
            #     self.n_dd = max(self.n_dd, len(v))
            #     print(self.n_dd)
        dd_list = np.zeros((len(input_list), self.n_dd), dtype=np.int64)
        for i,t in enumerate(time_list[: self.args.n_visit]):
            if int(t) not in visit_dict:
                continue
            id_list = visit_dict[int(t)]
            if len(id_list):
                id_list = np.array(id_list, dtype=np.int64)
                if len(id_list) > self.n_dd:
                    np.random.shuffle(id_list)
                    dd_list[i] = id_list[- self.n_dd:]
                else:
                    dd_list[i][:len(id_list)] = id_list

        # assert pre_time_list.max() < 256
        # assert post_time_list.max() < 256
        assert pre_time_list.min() >= 0
        assert post_time_list.min() >= 0
        pre_time_list[pre_time_list>200] = 200
        post_time_list[post_time_list>200] = 200
        assert len(mask_list[0]) == self.args.output_size
        assert len(mask_list[0]) == len(pre_input_list[0])

        # print(input_list.shape)
        return torch.from_numpy(input_list), torch.from_numpy(output_list), torch.from_numpy(mask_list), input_file,\
                torch.from_numpy(pre_input_list), torch.from_numpy(pre_time_list), torch.from_numpy(post_input_list), \
                torch.from_numpy(post_time_list), torch.from_numpy(dd_list)

    def pre_filling(self, input_data):
        valid_list = []
        input_value = []
        for line in input_data:
            values = line.strip().split(',')
            input_value.append(values)
            valid = []
            for iv,v in enumerate(values):
                if v.strip() not in ['', 'NA']:
                    valid.append(1)
                else:
                    valid.append(0)
            valid_list.append(valid)
        valid_list = np.array(valid_list)
        valid_list[0] = 0

        pre_filled_data = [x[:1] for x in input_value]
        pre_filled_data[0] = input_value[0]
        for i in range(1, valid_list.shape[1]):
            valid = valid_list[:, i]
            indices = np.where(valid > 0)[0]
            if len(indices):
                mean = np.mean([float(input_value[id][i]) for id in indices])
                first = indices[0]
            else:
                mean = 0
                first = 10000

            if self.args.model == 'mean':
                value_list = self.feature_value_dict[self.args.name_list[i - 1]]
                mean = value_list[int(len(value_list)/2)]

            last_value = mean
            for i_line in range(1, valid_list.shape[0]):
                if valid_list[i_line, i]:
                    pre_filled_data[i_line].append(input_value[i_line][i])
                    last_value = input_value[i_line][i]
                else:
                    pre_filled_data[i_line].append(str(last_value))
        new_input_data = [','.join(x) for x in pre_filled_data]
        return new_input_data



    def get_brnn_item(self, idx):
        input_file = self.files[idx]
        output_file = input_file.replace('with_missing', 'groundtruth')

        with open(output_file) as f:
            output_data = f.read().strip().split('\n')
        with open(input_file) as f:
            input_data = f.read().strip().split('\n')


        valid_list = []
        input_value = []
        for line in input_data:
            values = line.strip().split(',')
            input_value.append(values)
            valid = []
            for iv,v in enumerate(values):
                if v.strip() not in ['', 'NA']:
                    valid.append(1)
                else:
                    valid.append(0)
            valid_list.append(valid)
        valid_list = np.array(valid_list)
        valid_list[0] = 0

        pre_filled_data = [x[:1] for x in input_value]
        pre_filled_data[0] = input_value[0]
        for i in range(1, valid_list.shape[1]):
            valid = valid_list[:, i]
            indices = np.where(valid > 0)[0]
            if len(indices):
                # mean.append(np.mean([float(input_value[id][i]) for id in indices]))
                # first.append(indices[0])
                mean = np.mean([float(input_value[id][i]) for id in indices])
                first = indices[0]
            else:
                mean = 0
                first = 10000

            if self.args.model in ['mean', 'mice']:
                value_list = self.feature_value_dict[self.args.name_list[i - 1]]
                mean = value_list[int(len(value_list)/2)]

            last_value = mean
            for i_line in range(1, valid_list.shape[0]):
                if valid_list[i_line, i]:
                    pre_filled_data[i_line].append(input_value[i_line][i])

                    last_value = input_value[i_line][i]
                    next_value = last_value
                    for j in range(i_line+1, len(valid_list)):
                        if valid_list[j, i]:
                            next_value = input_value[j][i]
                            break
                    # mean = (float(last_value) + float(next_value)) / 2
                    mean = last_value
                else:
                    pre_filled_data[i_line].append(mean)


        input_list, output_list, mask_list = [], [], []
        feat_list = input_data[0].strip().split(',')
        for i_line in range(1, self.args.n_visit+1):
            input = []
            output = []
            mask = []
            if i_line >= len(input_data):
                i_line = len(input_data) - 1
                if i_line == len(input_data):
                    output_line = ['' for _ in output_line]
            else:
                input_line = input_data[i_line].strip().split(',')
                output_line = output_data[i_line].strip().split(',')

            assert len(valid_list) == len(input_data)

            for i_feat in range(1, len(feat_list)):
                iv = input_line[i_feat]
                ov = output_line[i_feat]


                if ov in ['NA', '']:
                    mask.append(-1)
                    output.append(0)
                    input.append(self.map_output(pre_filled_data[i_line][i_feat], feat_list, i_feat))
                elif valid_list[i_line, i_feat]:
                    mask.append(0)
                    output.append(self.map_output(ov, feat_list, i_feat))
                    input.append(output[-1])
                else:
                    mask.append(1)
                    output.append(self.map_output(ov, feat_list, i_feat))
                    input.append(self.map_output(pre_filled_data[i_line][i_feat], feat_list, i_feat))
            input_list.append(input)
            output_list.append(output)
            mask_list.append(mask)
        input_list = np.array(input_list, dtype=np.float32)
        output_list = np.array(output_list, dtype=np.float32)
        mask_list = np.array(mask_list, dtype=np.int64)
        return torch.from_numpy(input_list), torch.from_numpy(output_list), torch.from_numpy(mask_list), input_file

    def get_detroit_item(self, idx):
        input_file = self.files[idx]
        output_file = input_file.replace('with_missing', 'groundtruth')

        with open(output_file) as f:
            output_data = f.read().strip().split('\n')
        with open(input_file) as f:
            input_data = f.read().strip().split('\n')

        valid_list = []
        input_value = []
        for line in input_data:
            values = line.strip().split(',')
            input_value.append(values)
            valid = []
            for iv,v in enumerate(values):
                if v.strip() not in ['', 'NA']:
                    valid.append(1)
                else:
                    valid.append(0)
            valid_list.append(valid)
        valid_list = np.array(valid_list)
        valid_list[0] = 0

        pre_filled_data = [x[:1] for x in input_value]
        pre_filled_data[0] = input_value[0]
        for i in range(1, valid_list.shape[1]):
            valid = valid_list[:, i]
            indices = np.where(valid > 0)[0]
            if len(indices):
                first = indices[0]
            else:
                first = -1

            for i_line in range(1, valid_list.shape[0]):
                if valid_list[i_line, i]:
                    pre_filled_data[i_line].append(input_value[i_line][i])
                    last_value = input_value[i_line][i]
                elif first >= 0:
                    for ni in indices:
                        if ni > i_line:
                            break
                    next_value = input_value[ni][i]
                    if i_line < first:
                        pre_filled_data[i_line].append(next_value)
                    else:
                        try:
                            pre_filled_data[i_line].append((float(last_value)+ float(next_value))/ 2.0)
                        except:
                            print(indices)
                            print(first)
                            print(i_line)
                        pre_filled_data[i_line].append((float(last_value)+ float(next_value))/ 2.0)
                else:
                    pre_filled_data[i_line].append(0)


        input_list, output_list, mask_list = [], [], []
        feat_list = input_data[0].strip().split(',')
        for i_line in range(1, self.args.n_visit+1):
            input = []
            output = []
            mask = []
            if i_line >= len(input_data):
                i_line = len(input_data) - 1
                if i_line == len(input_data):
                    output_line = ['' for _ in output_line]
            else:
                input_line = input_data[i_line].strip().split(',')
                output_line = output_data[i_line].strip().split(',')

            assert len(valid_list) == len(input_data)

            for i_feat in range(1, len(feat_list)):
                iv = input_line[i_feat]
                ov = output_line[i_feat]


                if ov in ['NA', '']:
                    mask.append(-1)
                    output.append(0)
                    input.append(self.map_output(pre_filled_data[i_line][i_feat], feat_list, i_feat))
                elif valid_list[i_line, i_feat]:
                    mask.append(0)
                    output.append(self.map_output(ov, feat_list, i_feat))
                    input.append(output[-1])
                else:
                    mask.append(1)
                    output.append(self.map_output(ov, feat_list, i_feat))
                    input.append(self.map_output(pre_filled_data[i_line][i_feat], feat_list, i_feat))
            input_list.append(input)
            output_list.append(output)
            mask_list.append(mask)

        # last and next 2 visits
        input_list = input_list[:2] + input_list + input_list[-2:]
        new_input = []
        for i in range(2, 2 + self.args.n_visit):
            input = []
            for j in range(i-2, i+3):
                input = input + input_list[j]
            new_input.append(input)
        input_list = new_input

        input_list = np.array(input_list, dtype=np.float32)
        output_list = np.array(output_list, dtype=np.float32)
        mask_list = np.array(mask_list, dtype=np.int64)
        return torch.from_numpy(input_list), torch.from_numpy(output_list), torch.from_numpy(mask_list), input_file



    def __getitem__(self, idx):
        if self.args.model in ['brnn', 'brits', 'mean', 'mice']:
            return self.get_brnn_item(idx)
        elif self.args.model == 'detroit':
            return self.get_detroit_item(idx)
        else:
            return self.get_mm_item(idx)

    def __len__(self):
        return len(self.files)




