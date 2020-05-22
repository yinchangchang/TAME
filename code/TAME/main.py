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


import torch
from torch.autograd import Variable
from torch.backends import cudnn
from torch.nn import DataParallel
from torch.utils.data import DataLoader

import data_loader
from models import tame
import myloss
import function

sys.path.append('../tools')
import parse, py_op

args = parse.args
args.hidden_size = args.rnn_size = args.embed_size 
if torch.cuda.is_available():
    args.gpu = 1
else:
    args.gpu = 0
if args.model != 'tame':
    args.use_ve = 0
    args.use_mm = 0
    args.use_ta = 0
if args.use_ve == 0:
    args.value_embedding = 'no'
print 'epochs,', args.epochs

def _cuda(tensor, is_tensor=True):
    if args.gpu:
        if is_tensor:
            return tensor.cuda(async=True)
        else:
            return tensor.cuda()
    else:
        return tensor

def get_lr(epoch):
    lr = args.lr
    return lr

    if epoch <= args.epochs * 0.5:
        lr = args.lr
    elif epoch <= args.epochs * 0.75:
        lr = 0.1 * args.lr
    elif epoch <= args.epochs * 0.9:
        lr = 0.01 * args.lr
    else:
        lr = 0.001 * args.lr
    return lr

def index_value(data):
    '''
    map data to index and value
    '''
    if args.use_ve == 0:
        data = Variable(_cuda(data)) # [bs, 250]
        return data
    data = data.numpy()
    index = data / (args.split_num + 1)
    value = data % (args.split_num + 1)
    index = Variable(_cuda(torch.from_numpy(index.astype(np.int64))))
    value = Variable(_cuda(torch.from_numpy(value.astype(np.int64))))
    return [index, value]

def train_eval(data_loader, net, loss, epoch, optimizer, best_metric, phase='train'):
    print(phase)
    lr = get_lr(epoch)
    if phase == 'train':
        net.train()
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    else:
        net.eval()

    loss_list, pred_list, label_list, mask_list = [], [], [], []
    feature_mm_dict = py_op.myreadjson(os.path.join(args.file_dir, args.dataset + '_feature_mm_dict.json'))
    for b, data_list in enumerate(tqdm(data_loader)):
        data, label, mask, files = data_list[:4]
        data = index_value(data)
        if args.model == 'tame':
            pre_input, pre_time, post_input, post_time, dd_list = data_list [4:9]
            pre_input = index_value(pre_input)
            post_input = index_value(post_input)
            pre_time = Variable(_cuda(pre_time))
            post_time = Variable(_cuda(post_time))
            dd_list = Variable(_cuda(dd_list))
            neib = [pre_input, pre_time, post_input, post_time]

        label = Variable(_cuda(label)) # [bs, 1]
        mask = Variable(_cuda(mask)) # [bs, 1]
        if args.dataset in ['MIMIC'] and args.model == 'tame' and args.use_mm:
            output = net(data, neib=neib, dd=dd_list, mask=mask) # [bs, 1]
        elif args.model == 'tame' and args.use_ta:
            output = net(data, neib=neib, mask=mask) # [bs, 1]
        else:
            output = net(data, mask=mask) # [bs, 1]

        if phase == 'test':
            folder = os.path.join(args.result_dir, args.dataset, 'imputation_result')
            output_data = output.data.cpu().numpy()
            mask_data = mask.data.cpu().numpy().max(2)
            for (icu_data, icu_mask, icu_file) in zip(output_data, mask_data, files):
                icu_file = os.path.join(folder, icu_file.split('/')[-1].replace('.csv', '.npy'))
                np.save(icu_file, icu_data)
                if args.dataset == 'MIMIC':
                    with open(os.path.join(args.data_dir, args.dataset, 'train_groundtruth', icu_file.split('/')[-1].replace('.npy', '.csv'))) as f:
                        init_data = f.read().strip().split('\n')
                    # print(icu_file)
                    wf = open(icu_file.replace('.npy', '.csv'), 'w')
                    wf.write(init_data[0] + '\n')
                    item_list = init_data[0].strip().split(',')
                    if len(init_data) <= args.n_visit:
                        try:
                            assert len(init_data) == (icu_mask >= 0).sum() + 1
                        except:
                            pass
                            # print(len(init_data))
                            # print(sum(icu_mask >= 0))
                            # print(icu_file)
                    for init_line, out_line in zip(init_data[1:], icu_data):
                        init_line = init_line.strip().split(',')
                        new_line = [init_line[0]]
                        # assert len(init_line) == len(out_line) + 1
                        for item, iv, ov in zip(item_list[1:], init_line[1:], out_line):
                            if iv.strip() not in ['', 'NA']:
                                new_line.append('{:4.4f}'.format(float(iv.strip())))
                            else:
                                minv, maxv = feature_mm_dict[item]
                                ov = ov * (maxv - minv) + minv
                                new_line.append('{:4.4f}'.format(ov))
                        new_line = ','.join(new_line)
                        wf.write(new_line + '\n')
                    wf.close()


        loss_output = loss(output, label, mask)
        pred_list.append(output.data.cpu().numpy())
        loss_list.append(loss_output.data.cpu().numpy())
        label_list.append(label.data.cpu().numpy())
        mask_list.append(mask.data.cpu().numpy())

        if phase == 'train':
            optimizer.zero_grad()
            loss_output.backward()
            optimizer.step()


    pred = np.concatenate(pred_list, 0)
    label = np.concatenate(label_list, 0)
    mask = np.concatenate(mask_list, 0)
    metric_list = function.compute_nRMSE(pred, label, mask)
    avg_loss = np.mean(loss_list)

    print('\nTrain Epoch %03d (lr %.5f)' % (epoch, lr))
    print('loss: {:3.4f} \t'.format(avg_loss))
    print('metric: {:s}'.format('\t'.join(['{:3.4f}'.format(m) for m in metric_list[:2]])))


    metric = metric_list[0]
    if phase == 'valid' and (best_metric[0] == 0 or best_metric[0] > metric):
        best_metric = [metric, epoch]
        function.save_model({'args': args, 'model': net, 'epoch':epoch, 'best_metric': best_metric})
    metric_list = metric_list[2:]
    name_list = args.name_list
    assert len(metric_list) == len(name_list) * 2
    s = args.model
    for i in range(len(metric_list)/2):
        name = name_list[i] + ''.join(['.' for _ in range(40 - len(name_list[i]))])
        print('{:s}{:3.4f}......{:3.4f}'.format(name, metric_list[2*i], metric_list[2*i+1]))
        s = s+ '  {:3.4f}'.format(metric_list[2*i])
    if phase != 'train':
        print('\t\t\t\t best epoch: {:d}     best MSE on missing value: {:3.4f} \t'.format(best_metric[1], best_metric[0])) 
        print(s)
    return best_metric


def main():

    assert args.dataset in ['DACMI', 'MIMIC']
    if args.dataset == 'MIMIC':
        args.n_ehr = len(py_op.myreadjson(os.path.join(args.data_dir, args.dataset, 'ehr_list.json')))
    args.name_list = py_op.myreadjson(os.path.join(args.file_dir, args.dataset+'_feature_list.json'))[1:]
    args.output_size = len(args.name_list)
    files = sorted(glob(os.path.join(args.data_dir, args.dataset, 'train_with_missing/*.csv')))
    data_splits = py_op.myreadjson(os.path.join(args.file_dir, args.dataset + '_splits.json'))
    train_files = [f for idx in [0, 1, 2, 3, 4, 5, 6] for f in data_splits[idx]]
    valid_files = [f for idx in [7] for f in data_splits[idx]]
    test_files = [f for idx in [8, 9] for f in data_splits[idx]]
    if args.phase == 'test':
        train_phase, valid_phase, test_phase, train_shuffle = 'test', 'test', 'test', False
    else:
        train_phase, valid_phase, test_phase, train_shuffle = 'train', 'valid', 'test', True
    train_dataset = data_loader.DataBowl(args, train_files, phase=train_phase)
    valid_dataset = data_loader.DataBowl(args, valid_files, phase=valid_phase)
    test_dataset = data_loader.DataBowl(args, test_files, phase=test_phase)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=train_shuffle, num_workers=args.workers, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
    args.vocab_size = (args.output_size + 2) * (1 + args.split_num) + 5

    if args.model == 'tame':
        net = tame.AutoEncoder(args)
    loss = myloss.MSELoss(args)

    net = _cuda(net, 0)
    loss = _cuda(loss, 0)

    best_metric= [0,0]
    start_epoch = 0

    if args.resume:
        p_dict = {'model': net}
        function.load_model(p_dict, args.resume)
        best_metric = p_dict['best_metric']
        start_epoch = p_dict['epoch'] + 1

    parameters_all = []
    for p in net.parameters():
        parameters_all.append(p)

    optimizer = torch.optim.Adam(parameters_all, args.lr)

    if args.phase == 'train':
        for epoch in range(start_epoch, args.epochs):
            print('start epoch :', epoch)
            train_eval(train_loader, net, loss, epoch, optimizer, best_metric)
            best_metric = train_eval(valid_loader, net, loss, epoch, optimizer, best_metric, phase='valid')
        print 'best metric', best_metric

    elif args.phase == 'test':
        folder = os.path.join(args.result_dir, args.dataset, 'imputation_result')
        os.system('rm -r ' + folder)
        os.system('mkdir ' + folder)

        train_eval(train_loader, net, loss, 0, optimizer, best_metric, 'test')
        train_eval(valid_loader, net, loss, 0, optimizer, best_metric, 'test')
        train_eval(test_loader, net, loss, 0, optimizer, best_metric, 'test')

if __name__ == '__main__':
    main()
