"""
Created on Thu Dec 22 17:09:14 2016

@author: ansohn
"""

from tpot import TPOT
import time
import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split

from mdr import MDR
#from sklearn.ensemble import RandomForestClassifier


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


X_train, X_test, y_train, y_test = train_test_split(features, labels, 
                                                    train_size=0.75, 
                                                    test_size=0.25,
                                                    stratify=labels)


# Expert Knowledge Filter & MDR
tpot = TPOT(generations=100, population_size=100, verbosity=2, expert_source=[load_musu])
t1 = time.time()
tpot.fit(X_train, y_train)
print(tpot.score(X_test, y_test))
t2 = time.time()
print("Time lapsed: ", t2 - t1)
