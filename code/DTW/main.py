# 

import sys
reload(sys)
sys.setdefaultencoding('utf8')

import os
import sys

sys.path.append('../tools')

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
n_variables = 27


def dist_func(s1, s2):
    s1 = s1.reshape((s1.shape[0], 1, s1.shape[1]))
    s2 = s2.reshape((1, s2.shape[0], s2.shape[1]))
    dist = np.abs(s1 - s2)
    dist = dist * dist
    dist = dist.mean(2)
    dist = np.sqrt(dist)
    return dist

def compute_dtw (dist_mat, path, h, hi, hj):
    path = dtw(dist_mat, path)
    h[hi, hj] = path[0]
    h[hj, hi] = path[0]
    return path[0]

def dtw(dist_mat, path, i=0, j=0):
    if path[i,j,0] > - inf:
        return  path[i,j]
    assert i < dist_mat.shape[0]
    assert j < dist_mat.shape[1]
    if i == dist_mat.shape[0] - 1 and j == dist_mat.shape[1] - 1:
        avg, dist, steps = 0, 0, 0
    elif i == dist_mat.shape[0] - 1:
        avg, dist, steps = dtw(dist_mat, path, i, j + 1)
    elif j == dist_mat.shape[1] - 1:
        avg, dist, steps = dtw(dist_mat, path, i+1, j)
    else:
        avg, dist, steps = dtw(dist_mat, path, i+1, j) 
        for x in [dtw(dist_mat, path, i, j+1),dtw(dist_mat, path, i+1, j+1)]:
            if avg > x[0]:
                avg, dist, steps = x

    return_dist = dist_mat[i, j]  + dist
    return_steps = steps + 1
    return_avg = return_dist / return_steps

    path[i,j] = return_avg, return_dist, return_steps
    return path[i,j]


def norm(m, ms):
    m = (m - ms[0]) / ms[1]
    return m


def compute_dist_mat():
    files = glob(os.path.join(args.result_dir, args.dataset, 'imputation_result/*.csv')) # [:100]
    feature_ms_dict = py_op.myreadjson(os.path.join(args.file_dir, args.dataset + '_feature_ms_dict.json'))
    subtyping_dir = os.path.join(args.result_dir, args.dataset, 'subtyping')
    py_op.mkdir(subtyping_dir)
    hadm_id_list = []
    mean_variables = []
    hadm_variable_dict = { }
    all_values = []

    for i_fi, fi in enumerate(tqdm(files)):
        hadm_id = fi.split('/')[-1].split('.')[0]
        hadm_data = []
        for i_line, line in enumerate(open(fi)):
            if i_line:
                line_data = line.strip().split(',')
                line_data = np.array([float(x) for x in line_data])
                if len(line_data) != n_variables + 1:
                    print(i_fi, fi)
                if line_data[0] < 0:
                    continue
                elif line_data[0] < 24:
                    hadm_data.append(line_data)
                else:
                    break
            else:
                head = line.strip().split(',')[1:]
                assert len(head) == n_variables

        values = np.array(hadm_data, dtype=np.float32)
        values = values[-24:]
        times = values[:, 0]
        values = values[:, 1:]

        assert len(values.shape) == 2
        assert values.shape[1] == n_variables

        hadm_variable_dict[hadm_id] = values
        hadm_id_list.append(hadm_id)
        all_values.append(values)

    all_values = np.concatenate(all_values, 0)
    ms = [all_values.mean(0), all_values.std(0)]


    hadm_dist_matrix = np.zeros((len(hadm_id_list), len(hadm_id_list))) - 1
    for i in tqdm(range(len(hadm_id_list))):
        hadm_dist_matrix[i,i] = 0
        for j in range(i+1, len(hadm_id_list)):
            if hadm_dist_matrix[i,j] >= 0 or i == j:
                continue
            s1 = hadm_variable_dict[hadm_id_list[i]]
            s2 = hadm_variable_dict[hadm_id_list[j]]
            s1 = norm(s1, ms)
            s2 = norm(s2, ms)
            dist_mat = dist_func(s1, s2)
            path = np.zeros([dist_mat.shape[0], dist_mat.shape[1], 3]) - inf - 1
            compute_dtw(dist_mat, path, hadm_dist_matrix, i, j)

    py_op.mywritejson(os.path.join(subtyping_dir, 'hadm_id_list.json'), hadm_id_list)
    np.save(os.path.join(subtyping_dir, 'hadm_dist_matrix.npy'), hadm_dist_matrix)


def main():
    compute_dist_mat()

if __name__ == '__main__':
    main()
