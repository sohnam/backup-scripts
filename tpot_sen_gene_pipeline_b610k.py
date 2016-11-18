from tpot import TPOT
import numpy as np
import pandas as pd
import time

from sklearn.cross_validation import train_test_split
#from sklearn.ensemble import RandomForestClassifier

input_data = '/home/ansohn/Python/data/Genomics_data/bladder-gwas/bladder_610k_imputation_final_filtered-sen.snp_cglformat.txt'
b610k = pd.read_csv(input_data, sep='\t')
phenotype = b610k['phenotype']
individuals = b610k.drop('phenotype', axis=1)


# BLADDER SCORES
#re1 = '/home/ansohn/Python/data/Genomics_data/bladder-scores/multisurf-scores-10-bladder_610k_imputation_final_filtered-sen.snp_cglformat.txt'
#re2 = '/home/ansohn/Python/data/Genomics_data/bladder-scores/relieff-scores-10-10-bladder_610k_imputation_final_filtered-sen.snp_cglformat.txt'
#re3 = '/home/ansohn/Python/data/Genomics_data/bladder-scores/surf-scores-10-bladder_610k_imputation_final_filtered-sen.snp_cglformat.txt'
#re4 = '/home/ansohn/Python/data/Genomics_data/bladder-scores/surfstar-scores-10-bladder_610k_imputation_final_filtered-sen.snp_cglformat.txt'
#
#load_re1 = pd.read_csv(re1, sep='\t', header=3)
#load_re2 = pd.read_csv(re2, sep='\t', header=3)
#load_re3 = pd.read_csv(re3, sep='\t', header=3)
#load_re4 = pd.read_csv(re4, sep='\t', header=3)
#
#load_ekf = [load_re1, load_re2, load_re3, load_re4]
#
#ek_sen = '/home/ansohn/Python/data/Genomics_data/bladder-scores/sen.snp.gene'
#ekf_sen = pd.read_csv(ek_sen, sep='\t')
#cluster_by_gene_ekf = list(ekf_sen['#snp'].groupby(ekf_sen['gene'])) 
#
#gene_list = list(set(ekf_sen['gene']))
#snp_list = list(set(ekf_sen['#snp']))
#b = pd.DataFrame(index=gene_list, columns=snp_list)
#
#for i in range(29):
#    for snp in list(cluster_by_gene_ekf[i][1]):
#        if snp in b.ix[cluster_by_gene_ekf[i][0]]:
#            (b.ix[cluster_by_gene_ekf[i][0]])[snp] = True
#    (b.ix[cluster_by_gene_ekf[i][0]]).fillna(False, inplace=True)
#    
#ekf_gene_partition = []
#for i in range(29):
#    ekf_gene_partition.append(b.iloc[i])
#
#ekf_gene_partition.extend(load_ekf)
#

X_train, X_test, y_train, y_test = train_test_split(individuals, phenotype,
                                                    train_size=0.75, 
                                                    test_size=0.25)


# EKF & MDR 
#tpot = TPOT(generations=500, population_size=350, verbosity=2, expert_source=load_ekf)
#tpot = TPOT(generations=500, population_size=350, verbosity=2, expert_source=ekf_gene_partition)
#t1 = time.time()
#tpot.fit(X_train, y_train)
#print(tpot.score(X_test, y_test))
#t2 = time.time()
#print("Time lapsed: ", t2 - t1)

# MDR Only
tpot = TPOT(generations=500, population_size=350, verbosity=2, expert_source=None)
t1 = time.time()
tpot.fit(X_train, y_train)
print(tpot.score(X_test, y_test))
t2 = time.time()
print("Time lapsed: ", t2 - t1)

# Random Forest
#clf = RandomForestClassifier(max_depth=10, max_features=len(X_train.columns), 
#                             n_estimators=1000, random_state=1)
#clf.fit(X_train, y_train)
#print(clf.score(X_test, y_test))



