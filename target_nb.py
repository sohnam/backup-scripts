# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 17:57:10 2016

@author: ansohn
"""

import numpy as np
import pandas as pd

from mdr import MDR
from sklearn.naive_bayes import GaussianNB
from sklearn.cross_validation import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline

a10_005h = '/home/ansohn/Python/data/tpot_mdr_gametes_data/a-10/Her-0.05/a_10s_2000her_0.05__maf_0.2_EDM-1_01.txt'
a10_01h = '/home/ansohn/Python/data/tpot_mdr_gametes_data/a-10/Her-0.1/a_10s_2000her_0.1__maf_0.2_EDM-1_01.txt'
a10_02h = '/home/ansohn/Python/data/tpot_mdr_gametes_data/a-10/Her-0.2/a_10s_2000her_0.2__maf_0.2_EDM-1_01.txt'
a10_04h = '/home/ansohn/Python/data/tpot_mdr_gametes_data/a-10/Her-0.4/a_10s_2000her_0.4__maf_0.2_EDM-1_01.txt'

a100_005h = '/home/ansohn/Python/data/tpot_mdr_gametes_data/a-100/Her-0.05/a_100s_2000her_0.05__maf_0.2_EDM-1_01.txt'
a100_01h = '/home/ansohn/Python/data/tpot_mdr_gametes_data/a-100/Her-0.1/a_100s_2000her_0.1__maf_0.2_EDM-1_01.txt'
a100_02h = '/home/ansohn/Python/data/tpot_mdr_gametes_data/a-100/Her-0.2/a_100s_2000her_0.2__maf_0.2_EDM-1_01.txt'
a100_04h = '/home/ansohn/Python/data/tpot_mdr_gametes_data/a-100/Her-0.4/a_100s_2000her_0.4__maf_0.2_EDM-1_01.txt'

a1000_005h = '/home/ansohn/Python/data/tpot_mdr_gametes_data/a-1000/Her-0.05/a_1000s_2000her_0.05__maf_0.2_EDM-1_01.txt'
a1000_01h = '/home/ansohn/Python/data/tpot_mdr_gametes_data/a-1000/Her-0.1/a_1000s_2000her_0.1__maf_0.2_EDM-1_01.txt'
a1000_02h = '/home/ansohn/Python/data/tpot_mdr_gametes_data/a-1000/Her-0.2/a_1000s_2000her_0.2__maf_0.2_EDM-1_01.txt'
a1000_04h = '/home/ansohn/Python/data/tpot_mdr_gametes_data/a-1000/Her-0.4/a_1000s_2000her_0.4__maf_0.2_EDM-1_01.txt'

a5000_005h = '/home/ansohn/Python/data/tpot_mdr_gametes_data/a-5000/Her-0.05/a_5000s_2000her_0.05__maf_0.2_EDM-1_01.txt'
a5000_01h = '/home/ansohn/Python/data/tpot_mdr_gametes_data/a-5000/Her-0.1/a_5000s_2000her_0.1__maf_0.2_EDM-1_01.txt'
a5000_02h = '/home/ansohn/Python/data/tpot_mdr_gametes_data/a-5000/Her-0.2/a_5000s_2000her_0.2__maf_0.2_EDM-1_01.txt'
a5000_04h = '/home/ansohn/Python/data/tpot_mdr_gametes_data/a-5000/Her-0.4/a_5000s_2000her_0.4__maf_0.2_EDM-1_01.txt'


gametes_all = [a10_005h, a10_01h, a10_02h, a10_04h,
               a100_005h, a100_01h, a100_02h, a100_04h,
               a1000_005h, a1000_01h, a1000_02h, a1000_04h,
               a5000_005h, a5000_01h, a5000_02h, a5000_04h]
               
dataset_names = ['a10_005h', 'a10_01h', 'a10_02h', 'a10_04h',
                 'a100_005h', 'a100_01h', 'a100_02h', 'a100_04h',
                 'a1000_005h', 'a1000_01h', 'a1000_02h', 'a1000_04h',
                 'a5000_005h', 'a5000_01h', 'a5000_02h', 'a5000_04h']

output_txt = '/home/ansohn/Python/venvs/mdr/gametes_logs/target_scores.txt'
with open(output_txt, 'w') as t1:
    for i in range(16):
#        print(dataset)
        dataset = gametes_all[i]
        dataset_name = dataset_names[i]
        load_dataset = pd.read_csv(dataset, sep='\t')
        phenotype = load_dataset['Class'].values
        individuals = load_dataset.drop('Class', axis=1)
        individuals = individuals[['M0P0', 'M0P1']].values
        
        for i in range(30):

            X_train, X_test, y_train, y_test = train_test_split(individuals, phenotype, 
                                                                train_size=0.75, 
                                                                test_size=0.25)
                                                        
            target_pipeline = MDR()
            target_pipeline.fit(X_train, y_train)
            
            t1.write('{}\t{}\tmdr-perfect\n'.format(dataset_name, target_pipeline.score(X_test, y_test)))



