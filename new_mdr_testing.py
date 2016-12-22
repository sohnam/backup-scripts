# -*- coding: utf-8 -*-
"""
Created on Thu Dec 22 11:37:41 2016

@author: ansohn
"""

import os, sys
import numpy as np
from mdr import MDR
from mdr.utils import n_way_models
import pandas as pd
from skrebate import ReliefF, SURF, MultiSURF, SURFstar
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import cross_val_score


input_data = '/home/ansohn/Downloads/VDR_Data.tsv'
input_data_raw = '/home/ansohn/Downloads/VDR_Data_text.txt'

#vdr_data = pd.read_csv(input_data, sep='\t')
#phenotype = vdr_data['class']
#phenotype = phenotype.astype('int32')
#individuals = vdr_data.drop('class', axis=1)
#
#X_train, X_test, y_train, y_test = train_test_split(individuals, phenotype, 
#                                                    train_size=0.75, 
#                                                    test_size=0.25,
#                                                    #random_state=2,
#                                                    stratify=phenotype.values)

#my_mdr = MDR()
#my_mdr.fit_transform(X_train.values, y_train.values)

#m1 = n_way_models(my_mdr, X_train.values, y_train.values, n=[3], feature_names=list(X_train.columns))
#print(m1)


data = pd.read_csv(input_data, sep='\t')
features = data.drop('class', axis=1).values
labels = data['class'].values

print(data.columns)

output_root = '/home/ansohn/Documents/Results/'

for fs in [ReliefF, SURF, MultiSURF, SURFstar]:
    fs = fs()
    fs.fit(features, labels)
    print(fs.__class__.__name__, fs.feature_importances_)
    output_filename = output_root + 'VDR_' + str(fs.__class__.__name__) + '.txt'
    with open(output_filename, 'w') as f:
        f.write("Gene\tScore\n")
        for n in range(0,19):
            f.write(str(data.columns[n]) + '\t' + str((fs.feature_importances_)[n]) + '\n')
        f.close()


