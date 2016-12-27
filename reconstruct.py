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
import copy
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
#from sklearn.naive_bayes import BernoulliNB
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.cross_validation import train_test_split

################################## CGEMS Data Set ###############################################
#input_data = '/home/ansohn/Python/data/cgems/CGEMS-prostate-cancer-data-only-genes-predict-aggressive.csv'
#
#cgems_prostate = pd.read_csv(input_data)
#labels = cgems_prostate['class']
#labels = phenotype.astype('int32')
#features = cgems_prostate.drop('class', axis=1)
#
#load_re1 = pd.read_csv('/home/ansohn/Python/data/cgems/output/multisurf-scores-10-CGEMS-prostate-cancer-data-only-genes-predict-aggressive.csv', 
#                       sep='\t', header=3)
#load_re2 = pd.read_csv('/home/ansohn/Python/data/cgems/output/relieff-scores-10-10-CGEMS-prostate-cancer-data-only-genes-predict-aggressive.csv', 
#                       sep='\t', header=3)
#load_re3 = pd.read_csv('/home/ansohn/Python/data/cgems/output/surf-scores-10-CGEMS-prostate-cancer-data-only-genes-predict-aggressive.csv', 
#                       sep='\t', header=3)
#load_re4 = pd.read_csv('/home/ansohn/Python/data/cgems/output/surfstar-scores-10-CGEMS-prostate-cancer-data-only-genes-predict-aggressive.csv', 
#                       sep='\t', header=3)
#load_ekf = [load_re1, load_re2, load_re3, load_re4]
#################################################################################################


################################## VDR Data Set #################################################
input_file = '/home/ansohn/Downloads/VDR_Data.tsv'
data = pd.read_csv(input_file, sep='\t')
features = data.drop('class', axis=1)
labels = data['class']

musu = '/home/ansohn/Documents/Results/VDR_MultiSURF.txt'
ref = '/home/ansohn/Documents/Results/VDR_ReliefF.txt'
su = '/home/ansohn/Documents/Results/VDR_SURF.txt'
sust = '/home/ansohn/Documents/Results/VDR_SURFstar.txt'

load_musu = pd.read_csv(musu, sep='\t', header=0)
load_ref = pd.read_csv(ref, sep='\t', header=0)
load_su = pd.read_csv(su, sep='\t', header=0)
load_sust = pd.read_csv(sust, sep='\t', header=0)

load_ekf = [load_musu, load_ref, load_su, load_sust]
##################################################################################################


X_train, X_test, y_train, y_test = train_test_split(features, labels, 
                                                    train_size=0.75, 
                                                    test_size=0.25,
                                                    #random_state=2,
                                                    stratify=labels.values)

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


def _ekf(input_df, ekf_index, k_best=10): # random.randint(2,5)

    if set(load_ekf[ekf_index]) in [set([True, False]), set([True]), set([False])]:
        ekf_source = np.array(load_ekf[ekf_index])
        ekf_subset = list(itertools.compress(input_df.columns.values, ekf_source))
    else:
#        ekf_source = (load_relieff[ekf_index].iloc[:k_best])['Gene']
        ekf_source = (load_ekf[ekf_index]).sort_values(['Score'], ascending=False)
        ekf_source = ekf_source[:k_best]
        
        ekf_subset = list(ekf_source['Gene'])

    return input_df.loc[:, ekf_subset].copy()
    

def n_way_models(mdr_instance, X, y, n=[2], feature_names=None):
    if feature_names is None:
        feature_names = list(range(X.shape[1]))

    for cur_n in n:
        for features in itertools.combinations(range(X.shape[1]), cur_n):
            mdr_model = copy.deepcopy(mdr_instance)
            mdr_model.fit(X[:, features], y)
            mdr_model_score = mdr_model.score(X[:, features], y)
            model_features = [feature_names[feature] for feature in features]
            yield mdr_model, mdr_model_score, model_features



xtr = _ekf(training_features, ekf_index=0)
xte = _ekf(testing_features, ekf_index=0)

#my_mdr_tr = MDR(tie_break_choice, default_label_choice)
#my_mdr_te = MDR(tie_break_choice, default_label_choice)

mymdr = MDR()
clf = GaussianNB()

n_way_results = []
n_way_features = []
for nw in range(2,4):
#            subset_features = np.random.choice(training_features.columns, nw, replace=False)
#            training_features = training_features[subset_features]
    m1 = n_way_models(mymdr, xtr.values, training_classes, n=[nw], 
                      feature_names=list(xtr.columns))
    m2 = list(m1)
    
    for i in range(0, len(m2)):
        n_way_results.append( (m2[i])[1] )
#        n_way_results = tuple(n_way_results)
        n_way_features.append( (m2[i])[2] )
#        n_way_features = tuple(n_way_features)

d1 = dict(zip(n_way_results, n_way_features))
max_val = max(d1.keys())
max_feat = list(v for k, v in d1.items() if k == max_val)[0]

xtr = xtr[max_feat]
xte = xte[max_feat]

#clf.fit(mymdr.transform(xtr.values), training_classes)
#print('ekf + mdr: ', clf.score(mymdr.transform(xte.values), testing_classes))
mymdr.fit(xtr.values, training_classes)
print('ekf + mdr: ', mymdr.score(xte.values, testing_classes))


#randex = random.randint(0,3)

#selector = SelectKBest(f_classif, k=5)

# Feature selection with EKF
#xtr = _ekf(training_features, ekf_index=2)
#xte = _ekf(testing_features, ekf_index=2)

##full_data_0 = _ekf(individuals, ekf_index=0)
##full_data_2 = _ekf(individuals, ekf_index=2)
#
#xtr_2 = selector.fit_transform(training_features, training_classes)
#xte_2 = selector.fit_transform(testing_features, testing_classes)
#
#
#my_mdr_tr = MDR(tie_break_choice, default_label_choice)
#my_mdr_te = MDR(tie_break_choice, default_label_choice)
#clf = GaussianNB()
#
#my_mdr_tr.fit_transform(xtr.values, training_classes)
#my_mdr_te.fit_transform(xte.values, testing_classes)

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

#clf.fit(my_mdr_tr.transform(xtr.values), training_classes)
#print('ekf + mdr: ', clf.score(my_mdr_tr.transform(xte.values), testing_classes))


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

