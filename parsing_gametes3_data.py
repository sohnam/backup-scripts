import random
from collections import Counter
from operator import itemgetter
import itertools
import numpy as np
import pandas as pd
from mdr import MDR
from sklearn.naive_bayes import GaussianNB
from sklearn.cross_validation import train_test_split


input_data = '/home/ansohn/Python/data/cgems/CGEMS-prostate-cancer-data-only-genes-predict-aggressive.csv'

## CGEMS EKF
re1 = '/home/ansohn/Python/data/cgems/output/multisurf-scores-10-CGEMS-prostate-cancer-data-only-genes-predict-aggressive.csv'
re2 = '/home/ansohn/Python/data/cgems/output/relieff-scores-10-10-CGEMS-prostate-cancer-data-only-genes-predict-aggressive.csv'
re3 = '/home/ansohn/Python/data/cgems/output/surf-scores-10-CGEMS-prostate-cancer-data-only-genes-predict-aggressive.csv'
re4 = '/home/ansohn/Python/data/cgems/output/surfstar-scores-10-CGEMS-prostate-cancer-data-only-genes-predict-aggressive.csv'

cgems_prostate = pd.read_csv(input_data)
phenotype = cgems_prostate['class']
phenotype = phenotype.astype('int32')
individuals = cgems_prostate.drop('class', axis=1)

#gametes_filename = '/home/ansohn/Python/data/tpot_mdr_gametes_data/a-100/Her-0.4/a_100s_2000her_0.4__maf_0.2_EDM-1_01.txt'
#re1 = '/home/ansohn/Python/data/tpot_mdr_gametes_data/loc2_filtered/a-100_0.4/multisurf-scores-10-a_100s_2000her_0.4__maf_0.2_EDM-1_01.txt'
#re2 = '/home/ansohn/Python/data/tpot_mdr_gametes_data/loc2_filtered/a-100_0.4/relieff-scores-10-10-a_100s_2000her_0.4__maf_0.2_EDM-1_01.txt'
#re3 = '/home/ansohn/Python/data/tpot_mdr_gametes_data/loc2_filtered/a-100_0.4/surf-scores-10-a_100s_2000her_0.4__maf_0.2_EDM-1_01.txt'
#re4 = '/home/ansohn/Python/data/tpot_mdr_gametes_data/loc2_filtered/a-100_0.4/surfstar-scores-10-a_100s_2000her_0.4__maf_0.2_EDM-1_01.txt'
#gametes = pd.read_csv(gametes_filename, sep='\t')
#phenotype = gametes['Class']
#individuals = gametes.drop('Class', axis=1)


load_re1 = pd.read_csv(re1, sep='\t', header=3)
load_re2 = pd.read_csv(re2, sep='\t', header=3)
load_re3 = pd.read_csv(re3, sep='\t', header=3)
load_re4 = pd.read_csv(re4, sep='\t', header=3)
load_relieff = [load_re1, load_re2, load_re3, load_re4]

#ek_sen = '/home/ansohn/Python/data/Genomics_data/bladder-gwas/sen.snp.gene'
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
#ekf_gene_partition.extend(load_relieff)

X_train, X_test, y_train, y_test = train_test_split(individuals, phenotype, 
                                                    train_size=0.75, 
                                                    test_size=0.25)

training_data = pd.DataFrame(X_train)
training_data['class'] = y_train
training_data['group'] = 'training'

testing_data = pd.DataFrame(X_test)
testing_data['class'] = 0
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


# RELIEFF & co

def _ekf(input_df, ekf_index, k_best=20): # random.randint(2,5)

    if set(load_relieff[ekf_index]) in [set([True, False]), set([True]), set([False])]:
        ekf_source = np.array(load_relieff[ekf_index])
        ekf_subset = list(itertools.compress(input_df.columns.values, ekf_source))
    else:
        ekf_source = (load_relieff[ekf_index].iloc[:k_best])['Gene']
        ekf_subset = list(ekf_source)

    return input_df.loc[:, ekf_subset].copy()

# GENE PARTITION LIST
#def _ekf(input_df, ekf_index=29, k_best=13):
#    
#    if set(ekf_gene_partition[ekf_index]) in [set([True, False]), set([True]), set([False])]:
#        ekf_source = np.array(ekf_gene_partition[ekf_index])
#        ekf_subset = list(itertools.compress(input_df.columns.values, ekf_source))  
#    else:
#        ekf_source = (ekf_gene_partition[ekf_index].iloc[:k_best])['Gene']
#        ekf_subset = list(ekf_source)
#
#    return input_df.loc[:, ekf_subset].copy()  


xtr = _ekf(training_features, ekf_index=0)
xte = _ekf(testing_features, ekf_index=0)

input_cols = xtr.columns
track_index = np.random.randint(1, len(input_cols))
cols_len = len(input_cols)

#if len(input_cols) > 8:
#    
#    index_1 = np.random.randint(1, len(input_cols) + 1)
#    index_2 = np.random.randint(1, len(input_cols) + 1)
#    index_3 = np.random.randint(1, len(input_cols) + 1)
#    index_4 = np.random.randint(1, len(input_cols) + 1)
#    index_5 = np.random.randint(1, len(input_cols) + 1)
#    
#    if track_index == 5:
#        mdr_subset = np.random.choice(list(input_cols), 5, replace=False)
#    elif track_index == 4:
#        mdr_subset = np.random.choice(list(input_cols), 4, replace=False)
#    elif track_index == 3:
#        mdr_subset = np.random.choice(list(input_cols), 3, replace=False)
#    elif track_index == 2:
#        mdr_subset = np.random.choice(list(input_cols), 2, replace=False)
#    elif track_index == 1:
#        mdr_subset = np.random.choice(list(input_cols), 1, replace=False)
#    
#    xtr = xtr[mdr_subset]
#    xte = xte[mdr_subset]


#if len(input_cols) > 8:
#
#    track_index = (track_index % 5) + 1    
#    
#    index_1 = index_1 % len(input_cols)            
#    index_2 = index_2 % len(input_cols)
#    index_3 = index_3 % len(input_cols)
#    index_4 = index_4 % len(input_cols)
#    index_5 = index_5 % len(input_cols)
#    
#    if track_index == 5:
#        mdr_subset = list(itemgetter([index_1, index_2, index_3, index_4, index_5])(input_cols))
#    elif track_index == 4:
#        mdr_subset = list(itemgetter([index_1, index_2, index_3, index_4])(input_cols))
#    elif track_index == 3:
#        mdr_subset = list(itemgetter([index_1, index_2, index_3])(input_cols))
#    elif track_index == 2:
#        mdr_subset = list(itemgetter([index_1, index_2])(input_cols))
#    elif track_index == 1: 
#        mdr_subset = list(itemgetter([index_1])(input_cols))
#    
#    xtr = xtr[mdr_subset]
#    xte = xte[mdr_subset]


if len(input_cols) > 5:
    track_index = (track_index % 5) + 1
    
    index_1 = np.random.randint(1, len(training_features.columns) + 1)
    index_2 = np.random.randint(1, len(training_features.columns) + 1)
    index_3 = np.random.randint(1, len(training_features.columns) + 1)
    index_4 = np.random.randint(1, len(training_features.columns) + 1)
    index_5 = np.random.randint(1, len(training_features.columns) + 1)
    
    indices = [index_1, index_2, index_3, index_4, index_5]    
    
    if any(index >= cols_len for index in indices):
        f1 = list(filter(lambda x: x > cols_len, indices))
        f2 = list(filter(lambda y: y == cols_len, indices))
        new_list = list(set(indices) - set(f1))
        new_list = list(set(new_list) - set(f2))
        for d in f2:
            d = d - 1
            new_list.extend([d])
        for c in f1:
            c = c % cols_len
            new_list.extend([c])
            
    if track_index == 5:
        mdr_subset = list(itemgetter(new_list)(input_cols))
    elif track_index == 4:
        mdr_subset = list(itemgetter(new_list[:4])(input_cols))
    elif track_index == 3:
        mdr_subset = list(itemgetter(new_list[:3])(input_cols))
    elif track_index == 2:
        mdr_subset = list(itemgetter(new_list[:2])(input_cols))
    elif track_index == 1: 
        mdr_subset = list(itemgetter(new_list[:1])(input_cols))

    xtr = xtr[mdr_subset]
    xte = xte[mdr_subset]


my_mdr = MDR(tie_break_choice, default_label_choice)
clf = GaussianNB()

my_mdr.fit(xtr.values, training_classes)
print(my_mdr.score(xte.values, testing_classes))

#clf.fit(xtr.values, training_classes)
#print(clf.score(xte.values, testing_classes))

#for i in range(4):
#    for c in range(2, 6):
#        xtr = _ekf(training_features, i, k_best=c)
#        xte = _ekf(testing_features, i, k_best=c)
#        
#        my_mdr = MDR(tie_break_choice, default_label_choice)
#        clf = GaussianNB()
#        
#        my_mdr.fit(xtr.values, training_classes)
#        print('{}\t{}\t{}'.format(i, c, my_mdr.score(xte.values, testing_classes))) 


#cl = 10
#index_1, index_2, index_3, index_4, index_5 = 3, 7, 9, 1, 21
#
#l1 = [index_1, index_2, index_3, index_4, index_5]
#
#if any(index > cl for index in l1):
#    f1 = list(filter(lambda x: x > cl, l1))
#    l1 = list(set(l1) - set(f1))
#    print(l1)
#    for i in f1:
#        i = i % 10
#        l1.extend([i])
#    print(l1)
#else:
#    print("All smaller")





