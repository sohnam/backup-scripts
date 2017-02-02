#from mdr import MDR
import time
import numpy as np
import pandas as pd
#from sklearn.model_selection import cross_val_score

input_file = '/home/ansohn/Python/data/gametes-data/loc2_filtered/a5000/Her04/a_5000s_2000her_0.4__maf_0.2_EDM-1_01.txt'
data = pd.read_csv(input_file, sep='\t')

t1 = time.time()

data = data.sample(frac=1)
features = data.drop('Class', axis=1)
pred_feat = features.ix[:, ['M0P0','M0P1']]
print(pred_feat.columns)
#pred_feat = pred_feat.values
#labels = data['Class'].values

#mymdr = MDR()
#mymdr.fit(pred_feat, labels)
#pred_scores = cross_val_score(mymdr, pred_feat, labels, cv=10)
#print('mdr_pred scores: ', np.mean(pred_scores))
             
t2 = time.time()

