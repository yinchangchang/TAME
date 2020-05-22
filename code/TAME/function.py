#!/usr/bin/env python
# coding=utf-8
import numpy as np

import os
import torch
from sklearn import metrics


def compute_nRMSE(pred, label, mask):
    '''
    same as 3dmice
    '''
    assert pred.shape == label.shape == mask.shape

    missing_indices = mask==1
    missing_pred = pred[missing_indices]
    missing_label = label[missing_indices]
    missing_rmse = np.sqrt(((missing_pred - missing_label) ** 2).mean())

    init_indices = mask==0
    init_pred = pred[init_indices]
    init_label = label[init_indices]
    init_rmse = np.sqrt(((init_pred - init_label) ** 2).mean())

    metric_list = [missing_rmse, init_rmse]
    for i in range(pred.shape[2]):
        apred = pred[:,:,i]
        alabel = label[:,:, i]
        amask = mask[:,:, i]

        mrmse, irmse = [], []
        for ip in range(len(apred)):
            ipred = apred[ip]
            ilabel = alabel[ip]
            imask = amask[ip]

            x = ilabel[imask>=0]
            if len(x) == 0:
                continue

            minv = ilabel[imask>=0].min()
            maxv = ilabel[imask>=0].max()
            if maxv == minv:
                continue

            init_indices = imask==0
            init_pred = ipred[init_indices]
            init_label = ilabel[init_indices]

            missing_indices = imask==1
            missing_pred = ipred[missing_indices]
            missing_label = ilabel[missing_indices]

            assert len(init_label) + len(missing_label) >= 2

            if len(init_pred) > 0:
                init_rmse = np.sqrt((((init_pred - init_label) / (maxv - minv)) ** 2).mean())
                irmse.append(init_rmse)

            if len(missing_pred) > 0:
                missing_rmse = np.sqrt((((missing_pred - missing_label)/ (maxv - minv)) ** 2).mean())
                mrmse.append(missing_rmse)

        metric_list.append(np.mean(mrmse))
        metric_list.append(np.mean(irmse))

    metric_list = np.array(metric_list)


    metric_list[0] = np.mean(metric_list[2:][::2])
    metric_list[1] = np.mean(metric_list[3:][::2])

    return metric_list


def save_model(p_dict, name='best.ckpt', folder=None):
    args = p_dict['args']
    if folder is None:
        folder = os.path.join(args.data_dir, args.dataset, 'models')
    # name = '{:s}-snm-{:d}-snr-{:d}-value-{:d}-trend-{:d}-cat-{:d}-lt-{:d}-size-{:d}-seed-{:d}-loss-{:s}-{:d}-{:s}'.format(args.task, args.split_num, args.split_nor, args.use_value, args.use_trend, args.use_cat, args.last_time, args.embed_size, args.seed, args.loss, args.time, name)
    # name = '{:s}-{:s}-{:d}-variables-{:d}{:d}{:d}-{:s}'.format(args.dataset, args.model, len(args.name_list), args.use_ta, args.use_ve, args.use_mm, name)
    if not os.path.exists(folder):
        os.mkdir(folder)
    model = p_dict['model']
    state_dict = model.state_dict()
    for key in state_dict.keys():
        state_dict[key] = state_dict[key].cpu()
    all_dict = {
            'epoch': p_dict['epoch'],
            'args': p_dict['args'],
            'best_metric': p_dict['best_metric'],
            'state_dict': state_dict 
            }
    torch.save(all_dict, os.path.join(folder, name))

def load_model(p_dict, model_file):
    all_dict = torch.load(model_file)
    p_dict['epoch'] = all_dict['epoch']
    # p_dict['args'] = all_dict['args']
    p_dict['best_metric'] = all_dict['best_metric']
    # for k,v in all_dict['state_dict'].items():
    #     p_dict['model_dict'][k].load_state_dict(all_dict['state_dict'][k])
    p_dict['model'].load_state_dict(all_dict['state_dict'])

def compute_auc(labels, preds):
    fpr, tpr, thr = metrics.roc_curve(labels, preds)
    return metrics.auc(fpr, tpr)
