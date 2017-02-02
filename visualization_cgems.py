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

experiment_data = pd.concat([ekf, mdr, xgb, logit])

dataset = ['CGEMS_Prostate']


plt.figure(figsize=(6, 6))


ax1 = sb.boxplot(x="Data", y="Balanced_Accuracy", hue="Experiment", data=experiment_data, palette="gray").set(ylim=(0.45, 0.65))


plt.tight_layout()
plt.savefig('/home/ansohn/Python/Results/cgems/cgems-comparison.pdf', bbox_inches='tight', dpi=300)





