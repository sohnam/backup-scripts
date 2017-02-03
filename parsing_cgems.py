#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 13 10:50:40 2017

@author: ansohn
"""

from tpot import TPOTClassifier
from tpot.metrics import balanced_accuracy
from sklearn.metrics import make_scorer
import time
import pandas as pd
import numpy as np

import xgboost as xgb

#import numpy as np
#from tpot.operators.selectors.ek_filter import EKF_Source
#from mdr import MDR
from sklearn.model_selection import cross_val_score


input_file = '/home/ansohn/data/CGEMS-data/CGEMS-prostate-cancer-data-only-genes-predict-aggressive.csv'
data = pd.read_csv(input_file)

#musu = '/home/ansohn/cgems/cgems_msurf.txt'
#ref = '/home/ansohn/cgems/cgems_relieff.txt'
#su = '/home/ansohn/cgems/cgems_surf.txt'
#sust = '/home/ansohn/cgems/cgems_surfstar.txt'
#
#
#load_ekf = [musu, ref, su, sust]


########### Expert Knowledge Filter & MDR
#tpot_mdr = TPOTClassifier(generations=200, population_size=200, 
#                          num_cv_folds=10, verbosity=3, expert_source=load_ekf, n_jobs=-1)
#tpot_mdr = TPOTClassifier(generations=250, population_size=250, num_cv_folds=10, verbosity=2)

t1 = time.time()

data = data.sample(frac=1)
features = data.drop('class', axis=1).values
labels = data['class'].values

#tpot_mdr.fit(features, labels)

#t2 = time.time()
#print("Time lapsed: ", t2 - t1)


########### XGBoost
xgbclf = xgb.XGBClassifier(max_depth=10, learning_rate=0.0001, 
                           min_child_weight=20, subsample=0.5, n_estimators=500)

xgb_scores = cross_val_score(xgbclf, features, labels, 
                             cv=10, scoring=make_scorer(balanced_accuracy))
print('XGB_cgems: ', np.mean(xgb_scores))


