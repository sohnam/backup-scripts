# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 14:03:21 2016

@author: ansohn
"""

import pandas as pd
import numpy as np

from sklearn.cross_validation import train_test_split

# Training
gei_training = pd.read_csv('/home/ansohn/Python/data/Genomics_data/Gei/gei-training.tsv', sep='\t')
gei_knn_imputed_training = pd.read_csv('/home/ansohn/Python/data/Genomics_data/Gei/gei-knn-imputed-training.tsv', sep='\t')

# Testing
gei_testing = pd.read_csv('/home/ansohn/Python/data/Genomics_data/Gei/gei-testing.tsv', sep='\t')
gei_knn_imputed_testing = pd.read_csv('/home/ansohn/Python/data/Genomics_data/Gei/gei-knn-imputed-testing.tsv', sep='\t')

# Concat
gei_train_test = pd.concat([gei_training, gei_testing])
gei_knn_imputed_train_test = pd.concat([gei_knn_imputed_training, gei_knn_imputed_testing])

# Phenotypes/statuses
phenotype_gei_train_test = gei_train_test['phenotype']
status_gei_knn_imputed_train_test = gei_knn_imputed_train_test['status']

# Individuals
individuals_gei_train_test = gei_train_test.drop(['Unnamed: 0', 'phenotype'], axis=1)
individuals_gei_knn_imputed_train_test = gei_knn_imputed_train_test.drop('status', axis=1)

del gei_training, gei_testing, gei_knn_imputed_training, gei_knn_imputed_testing, gei_train_test, gei_knn_imputed_train_test


# gei_train_test
X_train, X_test, y_train, y_test = train_test_split(individuals_gei_train_test, phenotype_gei_train_test,
                                                    train_size=0.75, test_size=0.25)

# gei_knn_imputed_train_test                                                    
#X_train, X_test, y_train, y_test = train_test_split(individuals_gei_knn_imputed_train_test, status_gei_knn_imputed_train_test,
#                                                    train_size=0.75, test_size=0.25)                                                    

del phenotype_gei_train_test, status_gei_knn_imputed_train_test, individuals_gei_train_test, individuals_gei_knn_imputed_train_test


# EKFs for gei_knn_imputed
#load_gei_knn_imputed = pd.read_csv('/home/ansohn/Python/data/Genomics_data/Gei/multisurf-turf-scores-10-20-gei-knn-imputed-training.tsv', sep='\t', header=2)
#load_gei_knn_imputed_10_10 = pd.read_csv('/home/ansohn/Python/data/Genomics_data/Gei/relieff-scores-10-10-gei-knn-imputed-training.tsv', sep='\t', header=2)
#load_ekf_gei_knn = [load_gei_knn_imputed, load_gei_knn_imputed_10_10]

# EKFs for gei_training
load_gei_training_multisurf20 = pd.read_csv('/home/ansohn/Python/data/Genomics_data/Gei/gei-scores/multisurf-scores-20-gei-training.tsv', sep='\t', header=2)
load_gei_training_20_20_multisurf = pd.read_csv('/home/ansohn/Python/data/Genomics_data/Gei/gei-scores/multisurf-turf-scores-20-20-gei-training.tsv', sep='\t', header=2)
load_gei_20_20_multisurf = pd.read_csv('/home/ansohn/Python/data/Genomics_data/Gei/gei-scores/multisurf-turf-scores-20-20-gei.csv', sep='\t', header=2)
load_gei_multisurf20 = pd.read_csv('/home/ansohn/Python/data/Genomics_data/Gei/gei-scores/multisurf-scores-20-gei.csv', sep='\t', header=2)
load_ekf_gei_training = [load_gei_training_multisurf20, load_gei_20_20_multisurf, load_gei_training_20_20_multisurf, load_gei_multisurf20]



#load_gei_multisurfstar200 = pd.read_csv('/home/ansohn/Python/data/Genomics_data/Gei/gei-scores/gei.csv_multisurfstar200.txt', sep='\t')
#load_gei_multisurfstar10pct200 = pd.read_csv('/home/ansohn/Python/data/Genomics_data/Gei/gei-scores/gei.csv_multisurfstarnturf10pct_200.txt', sep='\t')
#load_gei_top200attrs = pd.read_csv('/home/ansohn/Python/data/Genomics_data/Gei/gei-scores/top_200_attrs-gei.csv', sep='\t')
#load_gei_turf_top200attrs = pd.read_csv('/home/ansohn/Python/data/Genomics_data/Gei/gei-scores/top_200_turf_attrs-gei.csv', sep='\t')












