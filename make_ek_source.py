#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 11 13:45:21 2017

@author: ansohn
"""

import os
import re
import sys
import glob

import pandas as pd
import numpy as np
from sklearn.pipeline import make_pipeline
from skrebate import ReliefF
from sklearn.model_selection import cross_val_score


genetic_data = pd.read_csv('/home/ansohn/Python/data/VDR-data/VDR_Data.tsv', sep='\t')
load_ref = pd.read_csv('/home/ansohn/Python/data/VDR-data/rearrange_ReliefF.txt', sep='\t', header=0)

features, labels = genetic_data.drop('class', axis=1), genetic_data['class'].values

clf = ReliefF(n_features_to_select=2, n_neighbors=100)

#clf = make_pipeline(ReliefF(n_features_to_select=2, n_neighbors=100),
#                    RandomForestClassifier(n_estimators=100))
#
#print(np.mean(cross_val_score(clf, features, labels)))






#input_txt = '/home/ansohn/Python/data/VDR-data/ReliefF.txt'
#input_txt = '/home/ansohn/Python/data/VDR-data/rearrange_ReliefF.txt'
#output_base = os.path.split(input_txt)
#oname, oext = re.split(r'\.(?!\d)', output_base[1])
#output_txt = output_base[0] + '/' + 'rearrange_' + oname + '.' + oext

#load_ref = pd.read_csv(input_txt, delimiter='\c')

#SNP = []
#f1 = open(input_txt, 'r')
#for line in f1:
#    if re.match('.*\brs*\b', line):
#        SNP.append(line)

#searchquery = 'rs'
#with open(input_txt) as f1:
#    with open(output_txt, 'a') as f2:
#        contents = f1.read()
#        for line in contents:
#            if re.match("(.*)rs(.*)", line): f2.write(line)



#for rf in glob.glob(input_txt):
#    with open(rf) as f1:
#        with open(output_txt, 'a') as f2:
#            contents = f1.read()
#            for i, line in enumerate(contents):
#                if line.startswith(searchquery):
#                    f2.write(line)



#def add_str_to_lines(f_name, str_to_add):
#    with open(f_name, "r") as f:
#        lines = f.readlines()
#        for index, line in enumerate(lines):
##             ADD BEFORE
##            lines[index] = str_to_add + line.strip() + "\n"
##             ADD AFTER
#            lines[index] =  line.strip() + str_to_add + "\n"
#
#    with open(f_name, "w") as f:
#        for line in lines:
#            f.write(line)
##
#if __name__ == "__main__":
#    add_str_to_lines(f_name=input_dir, str_to_add=str_to_add)

