#%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb
import pandas as pd
from glob import glob
from collections import defaultdict

sb.set_style('white')

## ekf_mdr
ekf = pd.read_csv('/home/ansohn/Python/Results/cgems/ekfs_only/cgems_ekfs_results.txt', sep='\t')
mdr = pd.read_csv('/home/ansohn/Python/Results/cgems/mdr_only/cgems_mdr_results.txt', sep='\t')
xgb = pd.read_csv('/home/ansohn/Python/Results/cgems/xgb/cgems_xgb_results.txt', sep='\t')
logit = pd.read_csv('/home/ansohn/Python/Results/cgems/logit/logit_cgems_results.txt', sep='\t')

experiment_data = pd.concat([mdr, xgb, logit, ekf])

cols = ["TPOT (MDR only)", "XGBoost", "Logistic Regression", "TPOT (MDR + EKF)"]

dataset = ['CGEMS_Prostate']


plt.figure(figsize=(6, 6))


ax1 = sb.boxplot(x="Data", y="Balanced_Accuracy", hue="Experiment", 
                 data=experiment_data, names=cols, palette="gray_r", 
                 notch=True)
ax1.set(xlabel='', ylabel='10-fold Balanced Accuracy')
vals = ax1.get_yticks()
ax1.set_yticklabels(['{}%'.format(int(round(x*100))) for x in vals])
ax1.set(xticklabels=[])

plt.ylim(0.49, 0.61)
plt.grid(True, axis='y', linestyle='--')
plt.tight_layout()
plt.savefig('/home/ansohn/Python/Results/cgems/cgems-comparison.pdf', bbox_inches='tight', dpi=300)





