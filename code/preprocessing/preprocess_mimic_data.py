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
args.data_dir = os.path.join(args.data_dir, args.dataset)

def time_to_second(t):
    t = str(t).replace('"', '')
    t = time.mktime(time.strptime(t,'%Y-%m-%d %H:%M:%S'))
    return int(t)

def map_ehr_id():
    print('start')
    ehr_count_dict = py_op.myreadjson(os.path.join(args.data_dir, 'ehr_count_dict.json'))
    ehr_list = [ehr for ehr,c in ehr_count_dict.items() if c > 100]
    ns = set('0123456789')
    print(ns)
    drug_list = [e for e in ehr_list if e[1] in ns]
    med_list = [e for e in ehr_list if e[1] not in ns]
    print(len(drug_list))
    print(len(med_list))
    py_op.mywritejson(os.path.join(args.data_dir, 'ehr_list.json'), ehr_list)


def generate_ehr_files():

    hadm_time_dict = py_op.myreadjson(os.path.join(args.data_dir, 'hadm_time_dict.json'))
    hadm_demo_dict = py_op.myreadjson(os.path.join(args.data_dir, 'hadm_demo_dict.json'))
    hadm_sid_dict = py_op.myreadjson(os.path.join(args.data_dir, 'hadm_sid_dict.json'))
    hadm_icd_dict = py_op.myreadjson(os.path.join(args.data_dir, 'hadm_icd_dict.json'))
    hadm_time_drug_dict = py_op.myreadjson(os.path.join(args.data_dir, 'hadm_time_drug_dict.json'))
    groundtruth_dir = os.path.join(args.data_dir, 'train_groundtruth')
    py_op.mkdir(groundtruth_dir)
    ehr_count_dict = dict()

    for hadm_id in hadm_sid_dict:

        time_drug_dict = hadm_time_drug_dict.get(hadm_id, { })
        icd_list = hadm_icd_dict.get(hadm_id, [])
        demo = hadm_demo_dict[hadm_id]
        demo[0] = demo[0] + '1'
        demo[1] = 'A' + str(int(demo[1] / 9))
        icd_demo = icd_list + demo

        for icd in icd_demo:
            ehr_count_dict[icd] = ehr_count_dict.get(icd, 0) + 1



        ehr_dict = { 'drug':{ }, 'icd_demo': icd_demo}

        for setime, drug_list in time_drug_dict.items():
            try:
                stime,etime = setime.split(' -- ')
                start_second = time_to_second(hadm_time_dict[hadm_id])
                stime = str((time_to_second(stime) - start_second) / 3600)
                etime = str((time_to_second(etime) - start_second) / 3600)
                setime = stime + ' -- ' + etime
                for drug in drug_list:
                    ehr_count_dict[drug] = ehr_count_dict.get(drug, 0) + 1
                ehr_dict['drug'][setime] = list(set(drug_list))
            except:
                pass


        py_op.mywritejson(os.path.join(groundtruth_dir, hadm_id + '.json'), ehr_dict)
        # break
    py_op.mywritejson(os.path.join(args.data_dir, 'ehr_count_dict.json'), ehr_count_dict)


def generate_demo():
    icu_hadm_dict = py_op.myreadjson('../../src/icu_hadm_dict.json')
    py_op.mywritejson(os.path.join(args.data_dir, 'icu_hadm_dict.json'), icu_hadm_dict)

    sid_demo_dict = dict()
    sid_hadm_dict = dict()
    for i_line, line in enumerate(open(os.path.join(args.mimic_dir, 'PATIENTS.csv'))):
        if i_line:
            data = line.split(',')
            sid = data[1]
            gender = data[2].replace('"', '')
            dob = data[3][:4]
            sid_demo_dict[sid] = [gender, int(dob)]
    py_op.mywritejson(os.path.join(args.data_dir, 'sid_demo_dict.json'), sid_demo_dict)

    hadm_sid_dict = dict()
    hadm_demo_dict = dict()
    hadm_time_dict = dict()
    for i_line, line in enumerate(open(os.path.join(args.mimic_dir, 'ICUSTAYS.csv'))):
        if i_line:
            line = line.replace('"', '')
            data = line.split(',')
            sid = data[1]
            hadm_id = data[2]
            icu_id = data[3]
            intime = data[-3]
            sid_hadm_dict[sid] = sid_hadm_dict.get(sid, []) + [hadm_id]
            if icu_id not in icu_hadm_dict:
                continue
            hadm_sid_dict[hadm_id] = sid
            gender = sid_demo_dict[sid][0]
            dob = sid_demo_dict[sid][1]
            age = int(intime[:4]) - dob
            if age < 18:
                print(age)
            assert age >= 18
            if age > 150:
                age = 90
            hadm_demo_dict[hadm_id] = [gender, age]
            hadm_time_dict[hadm_id] = intime
    py_op.mywritejson(os.path.join(args.data_dir, 'hadm_demo_dict.json'), hadm_demo_dict)
    py_op.mywritejson(os.path.join(args.data_dir, 'hadm_time_dict.json'), hadm_time_dict)
    py_op.mywritejson(os.path.join(args.data_dir, 'sid_hadm_dict.json'), sid_hadm_dict)
    py_op.mywritejson(os.path.join(args.data_dir, 'hadm_sid_dict.json'), hadm_sid_dict)

def generate_diagnosis_data():
    sid_hadm_dict = py_op.myreadjson(os.path.join(args.data_dir, 'sid_hadm_dict.json') )
    hadm_sid_dict = py_op.myreadjson(os.path.join(args.data_dir, 'hadm_sid_dict.json'))

    hadm_map_dict = dict()
    for hadm in hadm_sid_dict:
        sid = hadm_sid_dict[hadm]
        hadm_list = sid_hadm_dict[sid]
        if len(hadm_list) > 1:
            hadm_list = sorted(hadm_list, key=lambda k:int(k))
            idx = hadm_list.index(hadm)
            if idx > 0:
                for h in hadm_list[:idx]:
                    if h not in hadm_map_dict:
                        hadm_map_dict[h] = []
                    hadm_map_dict[h].append(hadm)

    hadm_icd_dict = dict()
    for i_line, line in enumerate(open(os.path.join(args.mimic_dir, 'DIAGNOSES_ICD.csv'))):
        if i_line:
            if i_line % 10000 == 0:
                print(i_line)
            line_data = [x.strip('"') for x in py_op.csv_split(line.strip())]
            ROW_ID, SUBJECT_ID, hadm_id, SEQ_NUM, icd = line_data
            if hadm_id in hadm_map_dict:
                for h in hadm_map_dict[hadm_id]:
                    if h not in hadm_icd_dict:
                        hadm_icd_dict[h] = []
                    hadm_icd_dict[h].append(icd)
    hadm_icd_dict = { h:list(set(icds)) for h, icds in hadm_icd_dict.items() }
    py_op.mywritejson(os.path.join(args.data_dir, 'hadm_icd_dict.json'), hadm_icd_dict)

def generate_drug_data():
    hadm_sid_dict = py_op.myreadjson(os.path.join(args.data_dir, 'hadm_sid_dict.json'))
    hadm_id_set = set(hadm_sid_dict)
    hadm_time_drug_dict = dict()
    for i_line, line in enumerate(open(os.path.join(args.mimic_dir, 'PRESCRIPTIONS.csv'))):
        if i_line:
            if i_line % 10000 == 0:
                print(i_line)
            line_data = [x.strip('"') for x in py_op.csv_split(line.strip())]
            _, SUBJECT_ID,hadm_id,_,startdate,enddate,_,drug,DRUG_NAME_POE,DRUG_NAME_GENERIC,FORMULARY_DRUG_CD,gsn,ndc,PROD_STRENGTH,DOSE_VAL_RX,DOSE_UNIT_RX,FORM_VAL_DISP,FORM_UNIT_DISP,ROUTE = line_data
            if len(hadm_id) and hadm_id in hadm_id_set:
                if hadm_id not in hadm_time_drug_dict:
                    hadm_time_drug_dict[hadm_id] = dict()
                time = startdate + ' -- ' + enddate
                if time not in hadm_time_drug_dict[hadm_id]:
                    hadm_time_drug_dict[hadm_id][time] = []
                hadm_time_drug_dict[hadm_id][time].append(drug)
                # hadm_time_drug_dict[hadm_id][time].append(ndc)
    py_op.mywritejson(os.path.join(args.data_dir, 'hadm_time_drug_dict.json'), hadm_time_drug_dict)




def main():
    generate_demo()

    generate_diagnosis_data()
    generate_drug_data()
    generate_ehr_files()
    map_ehr_id()

if __name__ == '__main__':
    main()
