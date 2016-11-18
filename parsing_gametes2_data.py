from tpot import TPOT
import time
import numpy as np
import pandas as pd

#from mdr import MDR
from sklearn.ensemble import RandomForestClassifier

from sklearn.cross_validation import train_test_split


gametes_filename = '/home/ansohn/Python/data/tpot_mdr_gametes_data/a-100/Her-0.4/a_100s_2000her_0.4__maf_0.2_EDM-1_01.txt'
re1 = '/home/ansohn/Python/data/tpot_mdr_gametes_data/loc2_filtered/a-100_0.4/multisurf-scores-10-a_100s_2000her_0.4__maf_0.2_EDM-1_01.txt'
re2 = '/home/ansohn/Python/data/tpot_mdr_gametes_data/loc2_filtered/a-100_0.4/relieff-scores-10-10-a_100s_2000her_0.4__maf_0.2_EDM-1_01.txt'
re3 = '/home/ansohn/Python/data/tpot_mdr_gametes_data/loc2_filtered/a-100_0.4/surf-scores-10-a_100s_2000her_0.4__maf_0.2_EDM-1_01.txt'
re4 = '/home/ansohn/Python/data/tpot_mdr_gametes_data/loc2_filtered/a-100_0.4/surfstar-scores-10-a_100s_2000her_0.4__maf_0.2_EDM-1_01.txt'

load_re1 = pd.read_csv(re1, sep='\t', header=3)
load_re2 = pd.read_csv(re2, sep='\t', header=3)
load_re3 = pd.read_csv(re3, sep='\t', header=3)
load_re4 = pd.read_csv(re4, sep='\t', header=3)

load_gametes = pd.read_csv(gametes_filename, sep='\t')

#load_re1 = pd.read_csv(re1, sep='\t', header=3).ix[2:]
#load_re2 = pd.read_csv(re2, sep='\t', header=3).ix[2:]
#load_re3 = pd.read_csv(re3, sep='\t', header=3).ix[2:]
#load_re4 = pd.read_csv(re4, sep='\t', header=3).ix[2:]

load_ekf = [load_re1, load_re2, load_re3, load_re4]

#cRe1 = load_re1.reindex(np.random.permutation(load_re1.index))
#cRe2 = load_re2.reindex(np.random.permutation(load_re1.index))
#cRe3 = load_re3.reindex(np.random.permutation(load_re2.index))
#cRe4 = load_re4.reindex(np.random.permutation(load_re3.index))
#load_ekf = [cRe1, cRe2, cRe3, cRe4]


phenotype = load_gametes['Class']
individuals = load_gametes.drop('Class', axis=1)

X_train, X_test, y_train, y_test = train_test_split(individuals, phenotype, 
                                                    train_size=0.75, 
                                                    test_size=0.25)


# Expert Knowledge Filter & MDR
tpot = TPOT(generations=200, population_size=200, verbosity=2, expert_source=load_ekf)
t1 = time.time()
tpot.fit(X_train, y_train)
print(tpot.score(X_test, y_test))
t2 = time.time()
print("Time lapsed: ", t2 - t1)

# MDR Only
#tpot = TPOT(generations=500, population_size=350, verbosity=2, expert_source=None)
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



