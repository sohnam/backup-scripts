# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 11:30:18 2016

@author: ansohn
"""

import random
from collections import Counter
import itertools
import pickle
import numpy as np
import pandas as pd
from mdr import MDR
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
#from sklearn.naive_bayes import BernoulliNB
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.cross_validation import train_test_split


input_data = '/home/ansohn/Python/data/cgems/CGEMS-prostate-cancer-data-only-genes-predict-aggressive.csv'

cgems_prostate = pd.read_csv(input_data)
phenotype = cgems_prostate['class']
phenotype = phenotype.astype('int32')
individuals = cgems_prostate.drop('class', axis=1)

load_re1 = pd.read_csv('/home/ansohn/Python/data/cgems/output/multisurf-scores-10-CGEMS-prostate-cancer-data-only-genes-predict-aggressive.csv', 
                       sep='\t', header=3)
load_re2 = pd.read_csv('/home/ansohn/Python/data/cgems/output/relieff-scores-10-10-CGEMS-prostate-cancer-data-only-genes-predict-aggressive.csv', 
                       sep='\t', header=3)
load_re3 = pd.read_csv('/home/ansohn/Python/data/cgems/output/surf-scores-10-CGEMS-prostate-cancer-data-only-genes-predict-aggressive.csv', 
                       sep='\t', header=3)
load_re4 = pd.read_csv('/home/ansohn/Python/data/cgems/output/surfstar-scores-10-CGEMS-prostate-cancer-data-only-genes-predict-aggressive.csv', 
                       sep='\t', header=3)
load_relieff = [load_re1, load_re2, load_re3, load_re4]

X_train, X_test, y_train, y_test = train_test_split(individuals, phenotype, 
                                                    train_size=0.75, 
                                                    test_size=0.25,
                                                    #random_state=2,
                                                    stratify=phenotype.values)

training_data = pd.DataFrame(X_train)
training_data['class'] = y_train
training_data['group'] = 'training'

testing_data = pd.DataFrame(X_test)
testing_data['class'] = y_test
testing_data['group'] = 'testing'

training_testing_data = pd.concat([training_data, testing_data])
most_frequent_class = Counter(X_train).most_common(1)[0][0]
training_testing_data['guess'] = most_frequent_class

non_feature_columns = ['class', 'group', 'guess']

training_features = training_testing_data.loc[training_testing_data['group'] == 'training'].drop(non_feature_columns, axis=1)
training_classes = training_testing_data.loc[training_testing_data['group'] == 'training', 'class'].values

testing_features = training_testing_data.loc[training_testing_data['group'] == 'testing'].drop(non_feature_columns, axis=1)
testing_classes = training_testing_data.loc[training_testing_data['group'] == 'testing', 'class'].values

tie_break = random.randint(1, 100)
default_label = random.randint(1, 100)
all_classes = sorted(training_testing_data['class'].unique())
tie_break_choice = all_classes[tie_break % len(all_classes)]
default_label_choice = all_classes[default_label % len(all_classes)]

tie_break_choice = 1
default_label_choice = 0


def _ekf(input_df, ekf_index, k_best=5): # random.randint(2,5)

    if set(load_relieff[ekf_index]) in [set([True, False]), set([True]), set([False])]:
        ekf_source = np.array(load_relieff[ekf_index])
        ekf_subset = list(itertools.compress(input_df.columns.values, ekf_source))
    else:
        ekf_source = (load_relieff[ekf_index].iloc[:k_best])['Gene']
        ekf_subset = list(ekf_source)

    return input_df.loc[:, ekf_subset].copy()

#randex = random.randint(0,3)

#selector = SelectKBest(f_classif, k=5)

# Feature selection with EKF
xtr = _ekf(training_features, ekf_index=2)
xte = _ekf(testing_features, ekf_index=2)

##full_data_0 = _ekf(individuals, ekf_index=0)
##full_data_2 = _ekf(individuals, ekf_index=2)
#
#xtr_2 = selector.fit_transform(training_features, training_classes)
#xte_2 = selector.fit_transform(testing_features, testing_classes)
#
#
my_mdr_tr = MDR(tie_break_choice, default_label_choice)
my_mdr_te = MDR(tie_break_choice, default_label_choice)
clf = GaussianNB()

my_mdr_tr.fit_transform(xtr.values, training_classes)
my_mdr_te.fit_transform(xte.values, testing_classes)

#my_mdr_tr.class_count_matrix
#my_mdr_te.class_count_matrix

#mtr = pd.DataFrame(list(my_mdr_tr.feature_map.items()))
#mte = pd.DataFrame(list(my_mdr_te.feature_map.items()))

############ NO TRAIN/TEST SPLIT ############
#my_mdr_0 = MDR(tie_break_choice, default_label_choice)
#my_mdr_2 = MDR(tie_break_choice, default_label_choice)
#
#my_mdr_0.fit(full_data_0.values, phenotype.values)
#my_mdr_2.fit(full_data_2.values, phenotype.values)
#
#m0 = pd.DataFrame(list(my_mdr_0.feature_map.items()))
#m2 = pd.DataFrame(list(my_mdr_2.feature_map.items()))
#
#m0.columns = ['Combinations', 'Phenotype']
#m2.columns = ['Combinations', 'Phenotype']
##############################################


clf.fit(my_mdr_tr.transform(xtr.values), training_classes)
print('ekf + mdr: ', clf.score(my_mdr_tr.transform(xte.values), testing_classes))


## Random Forest
#clf = RandomForestClassifier(max_depth=5, max_features=len(training_features.columns), 
#                             n_estimators=1000)
#clf.fit(training_features, training_classes)
#print(clf.score(testing_features, testing_classes))
                             
#clf.fit(individuals, phenotype)
#
#importances = pd.DataFrame({'feature':training_features.columns,'score':np.round(clf.feature_importances_,3)})
#importances = importances.sort_values('score',ascending=False).set_index('feature')
#print(importances.head(10))

#
#clf.fit(xtr_2, training_classes)
#print('SelectKBset: ', clf.score(xte_2, testing_classes))

#df_tr = list(itertools.filterfalse(lambda x: x in my_mdr_te.feature_map, my_mdr_tr.feature_map))
#df_te = list(itertools.filterfalse(lambda x: x in my_mdr_tr.feature_map, my_mdr_te.feature_map))
#d = {'df_tr': df_tr, 'df_te': df_te}

#d0 = list()
#d2 = 

#f0 = open('surf_tr_counts.pkl', 'wb')
#pickle.dump(dict(my_mdr_tr.class_count_matrix), f0, protocol=pickle.HIGHEST_PROTOCOL)
#f0.close()
#
#f2 = open('surf_te_counts.pkl', 'wb')
#pickle.dump(dict(my_mdr_te.class_count_matrix), f2, protocol=pickle.HIGHEST_PROTOCOL)
#f2.close()

