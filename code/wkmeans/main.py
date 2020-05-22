# coding=utf8

import sys
reload(sys)
sys.setdefaultencoding('utf8')

import os
import sys

sys.path.append('../tools')
sys.path.append('../imputation')

import copy
import time
import numpy as np
from sklearn import metrics
from sklearn.cluster import spectral_clustering
import random
import json
from glob import glob
from collections import OrderedDict
from tqdm import tqdm
from multiprocessing import Process, Pool
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


import parse, py_op

args = parse.args
inf = 100000000.0

def compute_weight(dist_mat, groups):
    weights = []
    for g in groups:
        dist_g = dist_mat[g][:, g]
        dist_avg = dist_g.mean(0)
        w = 1 / (1 + np.exp(dist_avg)) 
        w = w / w.sum()
        weights.append(w)
    return weights


def wkmeans_epoch(dist_mat, groups):

    assert dist_mat.min() >= 0
    weights = compute_weight(dist_mat, groups)

    cluster_dist = []
    for ig,g in enumerate(groups):
        dist = dist_mat[g]
        w = weights[ig]
        dist_avg = np.dot(w, dist)
        cluster_dist.append(dist_avg)

    new_groups = [[] for _ in groups]
    for i in range(len(dist_mat)):
        dist_i = [d[i] for d in cluster_dist]
        mind = min(dist_i)
        new_groups[dist_i.index(mind)].append(i)

    groups = new_groups
    return groups

def wkmeans(n_cluster):
    subtyping_dir = os.path.join(args.result_dir, args.dataset, 'subtyping')
    hadm_id_list = py_op.myreadjson(os.path.join(subtyping_dir, 'hadm_id_list.json'))
    hadm_dist_matrix = np.load(os.path.join(subtyping_dir, 'hadm_dist_matrix.npy'))
    assert len(hadm_dist_matrix) == len(hadm_id_list)

    # initialization
    indices = range(len(hadm_id_list))
    np.random.shuffle(indices)
    init_groups = [indices[i*10: i*10 + 10] for i in range(n_cluster)]

    groups = init_groups
    for epoch in range(100):
        groups = wkmeans_epoch(hadm_dist_matrix, groups)
        print([len(g) for g in groups])
        if epoch and epoch % 10 == 0:
            cluster_results = []
            for g in groups:
                cluster_results.append([hadm_id_list[i] for i in g])
            py_op.mywritejson(os.path.join(subtyping_dir, 'cluster_results.json'), cluster_results)

def main():
    wkmeans(args.nc)


if __name__ == '__main__':
    main()
