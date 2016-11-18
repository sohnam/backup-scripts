# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 10:40:25 2016

@author: ansohn
"""

from tpot import TPOT
import numpy as np
import pandas as pd
import time

from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier


input_data = '/home/ansohn/Python/data/cgems/CGEMS-prostate-cancer-data-only-genes-predict-aggressive.csv'
cgems_prostate = pd.read_csv(input_data)
phenotype = cgems_prostate['class']
phenotype = phenotype.astype('int32')
individuals = cgems_prostate.drop('class', axis=1)

# CGEMS EKF
re1 = '/home/ansohn/Python/data/cgems/output/multisurf-scores-10-CGEMS-prostate-cancer-data-only-genes-predict-aggressive.csv'
re2 = '/home/ansohn/Python/data/cgems/output/relieff-scores-10-10-CGEMS-prostate-cancer-data-only-genes-predict-aggressive.csv'
re3 = '/home/ansohn/Python/data/cgems/output/surf-scores-10-CGEMS-prostate-cancer-data-only-genes-predict-aggressive.csv'
re4 = '/home/ansohn/Python/data/cgems/output/surfstar-scores-10-CGEMS-prostate-cancer-data-only-genes-predict-aggressive.csv'

load_re1 = pd.read_csv(re1, sep='\t', header=3)
load_re2 = pd.read_csv(re2, sep='\t', header=3)
load_re3 = pd.read_csv(re3, sep='\t', header=3)
load_re4 = pd.read_csv(re4, sep='\t', header=3)

load_ekf = [load_re1, load_re2, load_re3, load_re4]

X_train, X_test, y_train, y_test = train_test_split(individuals, phenotype,
                                                    train_size=0.75, 
                                                    test_size=0.25)

# EKF & MDR 
tpot = TPOT(generations=500, population_size=500, verbosity=2, expert_source=load_ekf)
#tpot = TPOT(generations=500, population_size=500, verbosity=2, expert_source=[load_re1])
t1 = time.time()
tpot.fit(X_train, y_train)
print(tpot.score(X_test, y_test))
t2 = time.time()
print("Time lapsed: ", t2 - t1)

# MDR Only
#tpot = TPOT(generations=400, population_size=300, verbosity=2, expert_source=None)
#t1 = time.time()
#tpot.fit(X_train, y_train)
#print(tpot.score(X_test, y_test))
#t2 = time.time()
#print("Time lapsed: ", t2 - t1)

# Random Forest
#clf = RandomForestClassifier(max_depth=5, max_features=len(X_train.columns), 
#                             n_estimators=1000)
#clf.fit(X_train, y_train)
#print(clf.score(X_test, y_test))



