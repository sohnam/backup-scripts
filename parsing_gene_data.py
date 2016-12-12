from tpot import TPOT
import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction import DictVectorizer as DV

bladder_610K = '/home/ansohn/Python/data/Genomics_data/bladder_610k_imputation_final_filtered-sen.snp_cglformat.txt'
bladder_1M = '/home/ansohn/Python/data/Genomics_data/bladder_1M_imputation_final_filtered-sen.snp_cglformat.txt'

sen_sep_gene = '/home/ansohn/Python/data/Genomics_data/sen.snp.gene'
sen_sep_gene_models = '/home/ansohn/Python/data/Genomics_data/sen.snp-gene.models'

#m1 = np.genfromtxt(bladder_610K, delimiter='\t', names=True).dtype.names
#load_bladder_610K = np.genfromtxt(bladder_610K, delimiter='\t')

load_bladder_610K = pd.read_csv(bladder_610K, sep='\t')
load_sen_gene = pd.read_csv(sen_sep_gene, sep='\t')

#load_gene_models = pd.read_csv(sen_sep_gene_models, sep='\t')

#return_genes = load_sen_gene['gene'].astype('category')
return_gene = set(load_sen_gene['gene'])
gene_list = list(return_gene)
return_snp = set(load_sen_gene['#snp'])
snp_list = list(return_snp)

clustered_genes = []
for gene in gene_list:
    ret_spec_gene = load_sen_gene[load_sen_gene.gene == gene]
    clustered_genes.append(ret_spec_gene)

snp = load_sen_gene['#snp'].as_matrix()
gene = load_sen_gene['gene'].as_matrix()

train_test_data = pd.DataFrame.to_dict(load_sen_gene, 'series')
snps = train_test_data['#snp']
genes = train_test_data['gene']

d_array = {}
for snp in snps:
    d_array[snp] = {}
    for i in range(len(clustered_genes)):
        if snp in (clustered_genes[i])['#snp'].values:
            d_array[snp][(clustered_genes[i])['gene'].values[0]] = 1.0
        else:
            d_array[snp][(clustered_genes[i])['gene'].values[0]] = 0.0

df = pd.DataFrame(d_array).T

phenotype = load_bladder_610K['phenotype']
individuals = load_bladder_610K.drop('phenotype', axis=1)

X_train, X_test, y_train, y_test = train_test_split(individuals, phenotype, 
                                                    train_size=0.75, 
                                                    test_size=0.25, 
                                                    random_state=42)

tpot = TPOT(generations=10, population_size=10, verbosity=2)
tpot.fit(X_train, y_train)
print(tpot.score(X_test, y_test))
tpot.export('tpot_sen_gene_pipeline_b610k.py')


