#%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb
import pandas as pd
from glob import glob
from collections import defaultdict

sb.set_style('white')

## ekf_mdr
ekf1 = pd.read_csv('/home/ansohn/Python/venvs/mdr/gametes_logs/ekf_mdr/a10_all.txt', sep='\t')
ekf2 = pd.read_csv('/home/ansohn/Python/venvs/mdr/gametes_logs/ekf_mdr/a100_all.txt', sep='\t')
ekf3 = pd.read_csv('/home/ansohn/Python/venvs/mdr/gametes_logs/ekf_mdr/a1000_all.txt', sep='\t')
ekf4 = pd.read_csv('/home/ansohn/Python/venvs/mdr/gametes_logs/ekf_mdr/a5000_all.txt', sep='\t')
ekf_all = pd.concat([ekf1, ekf2, ekf3, ekf4])

## mdr_only
mdr1 = pd.read_csv('/home/ansohn/Python/venvs/mdr/gametes_logs/mdr_only/a10_all.txt', sep='\t')
mdr2 = pd.read_csv('/home/ansohn/Python/venvs/mdr/gametes_logs/mdr_only/a100_all.txt', sep='\t')
mdr3 = pd.read_csv('/home/ansohn/Python/venvs/mdr/gametes_logs/mdr_only/a1000_all.txt', sep='\t')
mdr4 = pd.read_csv('/home/ansohn/Python/venvs/mdr/gametes_logs/mdr_only/a5000_all.txt', sep='\t')
mdr_all = pd.concat([mdr1, mdr2, mdr3, mdr4])

## rf_only
rf1 = pd.read_csv('/home/ansohn/Python/venvs/mdr/gametes_logs/rf_only/a10_all.txt', sep='\t')
rf2 = pd.read_csv('/home/ansohn/Python/venvs/mdr/gametes_logs/rf_only/a100_all.txt', sep='\t')
rf3 = pd.read_csv('/home/ansohn/Python/venvs/mdr/gametes_logs/rf_only/a1000_all.txt', sep='\t')
rf4 = pd.read_csv('/home/ansohn/Python/venvs/mdr/gametes_logs/rf_only/a5000_all.txt', sep='\t')
rf_all = pd.concat([rf1, rf2, rf3, rf4])

## ekf_corrupt
#cor1 = pd.read_csv('/home/ansohn/Python/venvs/mdr/gametes_logs/ekf_corrupt/a10_all.txt', sep='\t')
#cor2 = pd.read_csv('/home/ansohn/Python/venvs/mdr/gametes_logs/ekf_corrupt/a100_all.txt', sep='\t')
#cor3 = pd.read_csv('/home/ansohn/Python/venvs/mdr/gametes_logs/ekf_corrupt/a1000_all.txt', sep='\t')
#cor_all = pd.concat([cor1, cor2, cor3])

# MDR target with P0/P1
mdr_perfect = pd.read_csv('/home/ansohn/Python/venvs/mdr/gametes_logs/target_scores.txt', sep='\t')

#experiment_data = pd.concat([ekf_all, mdr_all, rf_all, cor_all, mdr_perfect])
experiment_data = pd.concat([ekf_all, mdr_all, rf_all, mdr_perfect])

dataset = ['a10_005h', 'a10_01h', 'a10_02h', 'a10_04h', 
           'a100_005h', 'a100_01h', 'a100_02h', 'a100_04h', 
           'a1000_005h', 'a1000_01h', 'a1000_02h', 'a1000_04h',
           'a5000_005h', 'a5000_01h', 'a5000_02h', 'a5000_04h']

#dataset = ['a10_005h', 'a10_01h', 'a10_02h', 'a10_04h', 
#           'a100_005h', 'a100_01h', 'a100_02h', 'a100_04h', 
#           'a1000_005h', 'a1000_01h', 'a1000_02h', 'a1000_04h']

#target = pd.read_csv('/home/ansohn/Python/venvs/mdr/gametes_logs/gametes_target.txt', sep='\t')


plt.figure(figsize=(12, 12))

group_num = 1
for experiment in dataset:
    group = experiment_data.loc[experiment_data['Dataset']==experiment]
#    print(group)
#    break
    ax = plt.subplot(4, 4, group_num)
    sb.boxplot(data=group, x='Experiment', y='Score',
               notch=True, palette=sb.color_palette('Paired'), #sb.light_palette('black'),
               flierprops=dict(marker='o', markersize=5, markerfacecolor='black'),
#               order=['mdr_only', 'ekf_corrupt', 'ekf_mdr', 'rf_only'])
               order=['mdr_only', 'rf_only', 'ekf_mdr', 'mdr-perfect'])
    #plt.title(experiment.split('_discrete')[0])
    plt.xlabel('')

    
    if group_num == 5:
        plt.ylabel('Balanced Accuracy', fontsize=14)
    else:
        plt.ylabel('')
    
    if group_num % 4 == 1:
        plt.yticks(np.arange(0.4, 1.01, 0.1),
                   ['{}%'.format(int(round(x * 100.0, 0))) for x in np.arange(0.4, 1.01, 0.1)],
                   fontsize=12)
    elif False:#group_num % 4 == 0:
        ax.yaxis.tick_right()
        plt.yticks(np.arange(0.4, 1.01, 0.1),
                   ['{}%'.format(int(round(x * 100.0, 0))) for x in np.arange(0.4, 1.01, 0.1)],
                   fontsize=12)
    else:
        plt.yticks(np.arange(0.4, 1.01, 0.1),
                   [])
    
    plt.xticks([])
    
    if group_num <= 4:
        ax.xaxis.set_label_position('top') 
        plt.xlabel('{}'.format(0.05 * 2 ** (group_num - 1)), fontsize=16)
    
    if group_num % 4 == 0:
        ax.yaxis.set_label_position('right')
        attr = ['10', '100', '1000', '5000']
#        attr = ['10', '100', '1000']
        plt.ylabel('{}'.format(attr[int((group_num/4) - 1)]), fontsize=16)
        
#    plt.axhline(y=target.loc[target['Dataset']==experiment]['Score'].values, color='r')
    
    plt.grid(True, axis='y', linestyle='--')
    plt.ylim(0.4, 1.0)
    group_num += 1


plt.tight_layout()
plt.savefig('tpot-gametes-comparison.pdf', bbox_inches='tight', dpi=300)





