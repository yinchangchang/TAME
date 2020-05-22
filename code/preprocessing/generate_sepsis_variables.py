
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


variable_map_dict = {
            # lab
            'WBC': 'wbc',
            'bun': 'bun',
            'sodium': 'sodium',
            'pt': 'pt',
            'INR': 'inr',
            'PTT': 'ptt',
            'platelet': 'platelet',
            'lactate' : 'lactate',
            'hemoglobin': 'hemoglobin',
            'glucose': 'glucose',
            'chloride': 'chloride',
            'creatinine': 'creatinine',
            'aniongap': 'aniongap',
            'bicarbonate': 'bicarbonate',

            # other lab
            'hematocrit': 'hematocrit',

            # used
            'heart rate': 'heartrate',
            'respiratory rate': 'resprate',
            'temperature': 'tempc',
            'meanbp': 'meanbp',
            'gcs': 'gcs_min',
            'urineoutput': 'urineoutput',
            'sysbp': 'sysbp',
            'diasbp': 'diasbp',
            'spo2': 'spo2',
            'Magnesium': '',

            'C-reactive protein': '',
            'bands': 'bands',
            }

item_id_dict = {
            'C-reactive protein': '50889',
            'Magnesium': '50960', 
            }

def time_to_second(t):
    t = str(t).replace('"', '')
    t = time.mktime(time.strptime(t,'%Y-%m-%d %H:%M:%S'))
    return int(t)

def select_records_of_variables_not_in_pivoted():
    count_dict = { v:0 for v in item_id_dict.values() }
    hadm_time_dict = py_op.myreadjson(os.path.join(args.data_dir, args.dataset, 'hadm_time_dict.json' ))
    wf = open(os.path.join(args.mimic_dir, 'sepsis_lab.csv'), 'w')
    for i_line, line in enumerate(open(os.path.join(args.mimic_dir, 'LABEVENTS.csv'))):
        if i_line:
            line_data = line.split(',')
            if len(line_data) == 0:
                continue
            hadm_id, item_id, ctime = line_data[2:5]
            value = line_data[5]
            if item_id in count_dict and hadm_id in hadm_time_dict:
                # print(line)
                if len(line_data) != 9:
                    print(line)
                # assert len(line_data) == 9
                count_dict[item_id] += 1
                wf.write(line)
        else:
            wf.write(line)
            continue
        if i_line % 10000 == 0:
            print(i_line)
    wf.close()



def generate_variables_not_in_pivoted():
    assert args.dataset == 'MIMIC'
    id_item_dict = { v:k for k,v in item_id_dict.items() }
    head = sorted(item_id_dict)
    count_dict = { v:0 for v in item_id_dict.values() }
    wf = open(os.path.join(args.mimic_dir, 'pivoted_add.csv'), 'w')
    wf.write(','.join(['hadm_id', 'charttime'] + head) + '\n')
    for i_line, line in enumerate(open(os.path.join(args.mimic_dir, 'sepsis_lab.csv'))):
        if i_line:
            line_data = py_op.csv_split(line)
            hadm_id, item_id, ctime = line_data[2:5]
            value = line_data[6]
            try:
                value = float(value)
                index = head.index(id_item_dict[item_id])
                new_line = [hadm_id, ctime] + ['' for _ in range(index)] + [str(value)] + ['' for _ in range(index, len(head)-1)]
                new_line = ','.join(new_line) + '\n'
                wf.write(new_line)
            except:
                continue
            count_dict[item_id] += 1
            last_time = ctime
        else:
            print(line)
    print(count_dict)

def merge_pivoted_data(csv_list):
    name_list = ['hadm_id', 'charttime']
    for k,v in variable_map_dict.items():
        if k not in ['age', 'gender']:
            if len(v):
                name_list.append(v)
            elif k in item_id_dict:
                name_list.append(k)
    name_index_dict = { name:id for id,name in enumerate(name_list) }

    hadm_time_dict = py_op.myreadjson(os.path.join(args.data_dir, args.dataset, 'hadm_time_dict.json' ))
    icu_hadm_dict = py_op.myreadjson(os.path.join(args.data_dir, args.dataset, 'icu_hadm_dict.json' ))
    merge_dir = os.path.join(args.data_dir, args.dataset, 'merge_pivoted')
    os.system('rm -r ' + merge_dir)
    os.system('mkdir ' + merge_dir)
    pivoted_dir =  os.path.join(args.result_dir, 'mimic/pivoted_sofa')
    py_op.mkdir(pivoted_dir)

    for fi in csv_list:
        print(fi)
        for i_line, line in enumerate(open(os.path.join(args.mimic_dir, fi))):
            if i_line:
                line_data = line.strip().split(',')
                if len(line_data) <= 0:
                    continue
                line_dict = dict()
                for iv, v in enumerate(line_data):
                    if len(v.strip()):
                        name = head[iv]
                        line_dict[name] = v

                if fi == 'pivoted_sofa.csv':
                    icu_id = line_dict.get('icustay_id', 'xxx')
                    if icu_id not in icu_hadm_dict:
                        continue
                    hadm_id = str(icu_hadm_dict[icu_id])
                    line_dict['hadm_id'] = hadm_id
                    line_dict['charttime'] = line_dict['starttime']


                hadm_id = line_dict.get('hadm_id', 'xxx')
                if hadm_id not in hadm_time_dict:
                    continue
                hadm_time = time_to_second(hadm_time_dict[hadm_id])
                now_time = time_to_second(line_dict['charttime'])
                delta_hour = int((now_time - hadm_time) / 3600)
                line_dict['charttime'] = str(delta_hour)

                if fi == 'pivoted_sofa.csv':
                    sofa_file = os.path.join(pivoted_dir, hadm_id + '.csv')
                    if not os.path.exists(sofa_file):
                        with open(sofa_file, 'w') as f:
                            f.write(sofa_head)
                    wf = open(sofa_file, 'a')
                    sofa_line = [str(delta_hour)] + line.split(',')[4:]
                    wf.write(','.join(sofa_line))
                    wf.close()


                assert 'hadm_id' in line_dict
                assert 'charttime' in line_dict
                new_line = []
                for name in name_list:
                    new_line.append(line_dict.get(name, ''))
                new_line = ','.join(new_line) + '\n'
                hadm_file = os.path.join(merge_dir, hadm_id + '.csv')
                if not os.path.exists(hadm_file):
                    with open(hadm_file, 'w') as f:
                        f.write(','.join(name_list) + '\n')
                wf = open(hadm_file, 'a')
                wf.write(new_line)
                wf.close()

            else:
                if fi == 'pivoted_sofa.csv':
                    sofa_head = ','.join(['time'] + line.replace('"', '').split(',')[4:])
                # "icustay_id","hr","starttime","endtime","pao2fio2ratio_novent","pao2fio2ratio_vent","rate_epinephrine","rate_norepinephrine","rate_dopamine","rate_dobutamine","meanbp_min","gcs_min","urineoutput","bilirubin_max","creatinine_max","platelet_min","respiration","coagulation","liver","cardiovascular","cns","renal","respiration_24hours","coagulation_24hours","liver_24hours","cardiovascular_24hours","cns_24hours","renal_24hours","sofa_24hours"
                
                
                head = line.replace('"', '').strip().split(',')
                head = [h.strip() for h in head]
                # print(line)
                for h in head:
                    if h not in  name_index_dict:
                        print(h)

def sort_pivoted_data():
    sort_dir = os.path.join(args.data_dir, args.dataset, 'sort_pivoted')
    os.system('rm -r ' + sort_dir)
    os.system('mkdir ' + sort_dir)
    merge_dir = os.path.join(args.data_dir, args.dataset, 'merge_pivoted')

    for i_fi, fi in enumerate(tqdm(os.listdir(merge_dir))):
        wf = open(os.path.join(sort_dir, fi), 'w')
        time_line_dict = dict()
        for i_line, line in enumerate(open(os.path.join(merge_dir, fi))):
            if i_line:
                line_data = line.strip().split(',')
                delta = 3
                ctime = delta * int(int(line_data[1]) / delta)
                if ctime not in time_line_dict:
                    time_line_dict[ctime] = []
                time_line_dict[ctime].append(line_data)
            else:
                line_data = line.split(',')[1:]
                line_data[0] = 'time'
                wf.write(','.join(line_data))
        for t in sorted(time_line_dict):
            line_list = time_line_dict[t]
            new_line = line_list[0]
            for line_data in line_list[1:]:
                for iv, v in enumerate(line_data):
                    if len(v.strip()):
                        new_line[iv] = v
            new_line = ','.join(new_line[1:]) + '\n'
            wf.write(new_line)
        wf.close()
    py_op.mkdir('../../data/MIMIC/train_groundtruth')
    py_op.mkdir('../../data/MIMIC/train_with_missing')
    os.system('rm ../../data/MIMIC/train_groundtruth/*.csv')
    os.system('cp ../../data/MIMIC/sort_pivoted/* ../../data/MIMIC/train_groundtruth/')

def generate_icu_mortality_dict(icustay_id_list):
    icu_mortality_dict = dict()
    for i_line, line in enumerate(open(os.path.join(args.mimic_dir, 'sepsis_mortality.csv'))):
        if i_line:
            if i_line % 10000 == 0:
                print(i_line)
            line_data = line.strip().split(',')
            icustay_id = line_data[0]
            icu_mortality_dict[icustay_id] = int(line_data[-1])
    py_op.mywritejson(os.path.join(args.data_dir, 'icu_mortality_dict.json'), icu_mortality_dict)


def generate_lab_missing_values():
    lab_files = glob(os.path.join(args.data_dir, args.dataset, 'train_groundtruth/*.csv'))
    os.system('rm -r {:s}/*'.format(os.path.join(args.data_dir, args.dataset, 'train_with_missing')))
    feat_count_dict = dict()
    line_count_dict = dict()
    n_full = 0
    for i_fi, fi in enumerate(tqdm(lab_files)):
        file_data = []
        valid_list = []
        last_data = [-10000]
        for i_line, line in enumerate(open(fi)):
            if i_line:
                data = line.strip().split(',')
                # print(data)
                # print(fi)
                # assert(int(data[0])) > -200
                # assert(int(data[0])) < 800
                if int(data[0]) < -24 or int(data[0]) >= 500: 
                    continue
                assert(data[0]) > -200
                valid = []
                for i in range(len(data)):
                    feat_count_dict[feat_list[i]][0] += 1
                    if data[i] in ['', 'NA']:
                        feat_count_dict[feat_list[i]][1] += 1
                        valid.append(0)
                    else:
                        valid.append(1)
                        vector[i] = 1

                if data[0] == last_data[0]:
                    for iv in range(len(data)):
                        if valid[iv]:
                            last_valid[iv] = valid[iv]
                            last_data[iv] = data[iv]
                    valid_list[-1] = last_valid
                    file_data[-1] = last_data
                else:
                    valid_list.append(valid)
                    assert int(data[0]) < 700
                    assert int(data[0]) > - 200
                    file_data.append(data)
                    last_data = data
                    last_valid = valid
            else:
                feat_list = line.strip().split(',')
                vector = [0 for _ in feat_list]
                for feat in feat_list:
                    if feat not in feat_count_dict:
                        feat_count_dict[feat] = [0, 0]

                valid_list.append([1 for _ in feat_list])
                file_data.append(feat_list)
        line_count_dict[i_line] = line_count_dict.get(i_line, 0) + 1

        vs = [0 for _ in file_data[0]]
        for data in file_data[1:]:
            for iv, v in enumerate(data):
                if v.strip() not in ['', 'NA']:
                    vs[iv] += 1
        if np.min(vs) >= 2:
            n_full +=1
        # continue


        # if len(file_data)< 15 or np.min(vs) < 2:
        if len(file_data)< 5 or sorted(vector)[2] < 1:
            os.system('rm -r ' + fi)
            # os.system('rm -r ' + fi.replace('groundtruth', 'with_missing'))
            # print('rm -r ' + fi.replace('groundtruth', 'with_missing'))
        else:
            for data in file_data[1:]:
                assert int(data[0]) > -200
            # write groundtruth data
            x = [','.join(line) for line in file_data]
            x = '\n'.join(x)
            with open(fi, 'w') as f:
                f.write(x)

            valid_list = np.array(valid_list)
            valid_list[0] = 0
            for i in range(1, valid_list.shape[1]):
                valid = valid_list[:, i]
                indices = np.where(valid > 0)[0]
                indices = sorted(indices)
                if len(indices) > 2:
                    indices = indices[1:-1]
                    np.random.shuffle(indices)
                    file_data[indices[0]][i] = ''
            # write groundtruth data
            x = [','.join(line) for line in file_data]
            x = '\n'.join(x)
            with open(fi.replace('groundtruth', 'with_missing'), 'w') as f:
                f.write(x)
    print(n_full)


def main():
    csv_list = ['pivoted_sofa.csv', 'pivoted_add.csv', 'pivoted_lab.csv', 'pivoted_vital.csv']
    select_records_of_variables_not_in_pivoted()
    generate_variables_not_in_pivoted()
    merge_pivoted_data(csv_list)
    sort_pivoted_data()
    generate_lab_missing_values()



if __name__ == '__main__':
    main()
