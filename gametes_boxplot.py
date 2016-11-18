import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#df1_ekf = pd.read_csv(a10s_ekf, sep="\t")
#df1_mdr = pd.read_csv(a10s_mdr, sep="\t")
#df1_combine = pd.concat([df1_ekf, df1_mdr])

sns.set_style("whitegrid")

#df2_ekf = pd.read_csv('/home/ansohn/Python/venvs/mdr/old_logs/ekf_mdr/a5000s.txt', sep="\t")
#df2_mdr = pd.read_csv('/home/ansohn/Python/venvs/mdr/old_logs/mdr_only/a5000s.txt', sep="\t")
#df2_rf = pd.read_csv('/home/ansohn/Python/venvs/mdr/old_logs/rf_only/a5000s.txt', sep="\t")
#df2_ekf_c = pd.read_csv('/home/ansohn/Python/venvs/mdr/gametes_logs/ekf_corrupt/a100_all.txt', sep="\t")

#df2_combine = pd.concat([df2_ekf, df2_mdr, df2_rf, df2_ekf_c])
#df2_combine = pd.concat([df2_ekf, df2_mdr, df2_rf])

#ax1 = sns.boxplot(x="Dataset", y="Score", hue="Experiment", data=df2_combine, palette="Set2").set(ylim=(0.40, 0.85))

# b610k
#df2_combine = pd.read_csv('/home/ansohn/Python/venvs/mdr/bladder_gwas_logs/b610k/b610k_all.txt', sep='\t')
#ax_b610 = sns.boxplot(x="Bladder_dataset", y="Score", hue="Algorithm", data=df2_combine, palette="Set2").set(ylim=(0.40, 0.65))

# b1M
#df2_combine = pd.read_csv('/home/ansohn/Python/venvs/mdr/bladder_gwas_logs/b1M/b1M_all.txt', sep='\t')
#ax_b1M = sns.boxplot(x="Bladder_dataset", y="Score", hue="Algorithm", data=df2_combine, palette="Set2").set(ylim=(0.40, 0.65))


# ekf_corrupt -- correct pipeline
#plt.figure(figsize=(8, 6))
#plt.xlabel('')
#plt.ylabel('Correct Pipeline (%)')
#
#corr_10 = pd.read_csv('/home/ansohn/Python/venvs/mdr/gametes_logs/ekf_corrupt/a10_success.txt', sep='\t')
#corr_100 = pd.read_csv('/home/ansohn/Python/venvs/mdr/gametes_logs/ekf_corrupt/a100_success.txt', sep='\t')
#corr_1000 = pd.read_csv('/home/ansohn/Python/venvs/mdr/gametes_logs/ekf_corrupt/a1000_success.txt', sep='\t')
#
## ekf_mdr -- correct pipeline
#ekf_mdr_10 = pd.read_csv('/home/ansohn/Python/venvs/mdr/gametes_logs/ekf_mdr/a10_success.txt', sep='\t')
#ekf_mdr_100 = pd.read_csv('/home/ansohn/Python/venvs/mdr/gametes_logs/ekf_mdr/a100_success.txt', sep='\t')
#ekf_mdr_1000 = pd.read_csv('/home/ansohn/Python/venvs/mdr/gametes_logs/ekf_mdr/a1000_success.txt', sep='\t')
#
#a10_success = pd.concat([corr_10, ekf_mdr_10])
#a100_success = pd.concat([corr_100, ekf_mdr_100])
#a1000_success = pd.concat([corr_1000, ekf_mdr_1000])

#ax = sns.barplot(x='Dataset', y='Correct Pipeline (%)', hue='Experiment', 
#                 data=a1000_success, palette=sns.color_palette('Paired'))
#ax.set(ylim=(0.0, 1.2))
#ax.set(xlabel='', ylabel='Correct Pipeline (%)')



cgems_all = pd.read_csv('/home/ansohn/Python/venvs/mdr/cgems_prostate/cgems_all.txt', sep='\t')
ax1 = sns.boxplot(x="Dataset", y="Score", hue="Experiment", data=cgems_all, palette="Set2").set(ylim=(0.40, 0.70))
