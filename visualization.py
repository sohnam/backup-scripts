#%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb
import pandas as pd
from glob import glob
from collections import defaultdict

sb.set_style('white')

## ekf_mdr
ekf1 = pd.read_csv('/home/ansohn/Python/Results/gametes/ekf_mdr/ekfmdr_a10_all.txt', sep='\t')
ekf2 = pd.read_csv('/home/ansohn/Python/Results/gametes/ekf_mdr/ekfmdr_a100_all.txt', sep='\t')
ekf3 = pd.read_csv('/home/ansohn/Python/Results/gametes/ekf_mdr/ekfmdr_a1000_all.txt', sep='\t')
ekf4 = pd.read_csv('/home/ansohn/Python/Results/gametes/ekf_mdr/ekfmdr_a5000_all.txt', sep='\t')
ekf_all = pd.concat([ekf1, ekf2, ekf3, ekf4])

## mdr_only
mdr1 = pd.read_csv('/home/ansohn/Python/Results/gametes/mdr_only/mdr_a10_all.txt', sep='\t')
mdr2 = pd.read_csv('/home/ansohn/Python/Results/gametes/mdr_only/mdr_a100_all.txt', sep='\t')
mdr3 = pd.read_csv('/home/ansohn/Python/Results/gametes/mdr_only/mdr_a1000_all.txt', sep='\t')
mdr4 = pd.read_csv('/home/ansohn/Python/Results/gametes/mdr_only/mdr_a5000_all.txt', sep='\t')
mdr_all = pd.concat([mdr1, mdr2, mdr3, mdr4])

## xgb_only
xgb1 = pd.read_csv('/home/ansohn/Python/Results/gametes/xgb/xgb_a10_all.txt', sep='\t')
xgb2 = pd.read_csv('/home/ansohn/Python/Results/gametes/xgb/xgb_a100_all.txt', sep='\t')
xgb3 = pd.read_csv('/home/ansohn/Python/Results/gametes/xgb/xgb_a1000_all.txt', sep='\t')
xgb4 = pd.read_csv('/home/ansohn/Python/Results/gametes/xgb/xgb_a5000_all.txt', sep='\t')
xgb_all = pd.concat([xgb1, xgb2, xgb3, xgb4])


# MDR target with P0/P1
pred1 = pd.read_csv('/home/ansohn/Python/Results/gametes/mdr_pred/mdr_pred_a10_all.txt', sep='\t')
pred2 = pd.read_csv('/home/ansohn/Python/Results/gametes/mdr_pred/mdr_pred_a100_all.txt', sep='\t')
pred3 = pd.read_csv('/home/ansohn/Python/Results/gametes/mdr_pred/mdr_pred_a1000_all.txt', sep='\t')
pred4 = pd.read_csv('/home/ansohn/Python/Results/gametes/mdr_pred/mdr_pred_a5000_all.txt', sep='\t')
pred_all = pd.concat([pred1, pred2, pred3, pred4])


# Logistic regression
lr1 = pd.read_csv('/home/ansohn/Python/Results/gametes/logit/logit_a10_all.txt', sep='\t')
lr2 = pd.read_csv('/home/ansohn/Python/Results/gametes/logit/logit_a100_all.txt', sep='\t')
lr3 = pd.read_csv('/home/ansohn/Python/Results/gametes/logit/logit_a1000_all.txt', sep='\t')
lr4 = pd.read_csv('/home/ansohn/Python/Results/gametes/logit/logit_a5000_all.txt', sep='\t')
lr_all = pd.concat([lr1, lr2, lr3, lr4])

experiment_data = pd.concat([mdr_all, xgb_all, lr_all, ekf_all, pred_all])

dataset = ['a10_005h', 'a10_01h', 'a10_02h', 'a10_04h', 
           'a100_005h', 'a100_01h', 'a100_02h', 'a100_04h', 
           'a1000_005h', 'a1000_01h', 'a1000_02h', 'a1000_04h',
           'a5000_005h', 'a5000_01h', 'a5000_02h', 'a5000_04h']

#target = pd.read_csv('/home/ansohn/Python/venvs/mdr/gametes_logs/gametes_target.txt', sep='\t')


plt.figure(figsize=(12, 12))

group_num = 1
for experiment in dataset:
    group = experiment_data.loc[experiment_data['Dataset']==experiment]
#    print(group)
#    break
    ax = plt.subplot(4, 4, group_num)
    sb.boxplot(data=group, x='Experiment', y='Score',
               notch=True, palette=sb.light_palette('black'),  #sb.color_palette('Paired'),
               flierprops=dict(marker='o', markersize=5, markerfacecolor='black'),
               order=['mdr_only', 'xgb_only', 'logit', 'ekf_mdr', 'mdr_pred'])
    #plt.title(experiment.split('_discrete')[0])
    plt.xlabel('')

    
    if group_num == 5:
        plt.ylabel('')  # , fontsize=14
    else:
        plt.ylabel('')
    
    if group_num % 4 == 1:
        plt.yticks(np.arange(0.4, 0.85, 0.1),
                   ['{}%'.format(int(round(x * 100.0, 0))) for x in np.arange(0.4, 0.85, 0.1)],
                   fontsize=12)
    elif False:#group_num % 4 == 0:
        ax.yaxis.tick_right()
        plt.yticks(np.arange(0.4, 0.85, 0.1),
                   ['{}%'.format(int(round(x * 100.0, 0))) for x in np.arange(0.4, 0.85, 0.1)],
                   fontsize=12)
    else:
        plt.yticks(np.arange(0.4, 0.85, 0.1),
                   [])
    
    plt.xticks([])
    
    if group_num <= 4:
        ax.xaxis.set_label_position('top') 
        plt.xlabel('{}'.format(0.05 * 2 ** (group_num - 1)), fontsize=16)
    
    if group_num % 4 == 0:
        ax.yaxis.set_label_position('right')
        attr = ['10', '100', '1000', '5000']
        plt.ylabel('{}'.format(attr[int((group_num/4) - 1)]), fontsize=16)
        
#    plt.axhline(y=target.loc[target['Dataset']==experiment]['Score'].values, color='r')
    
    plt.grid(True, axis='y', linestyle='--')
    plt.ylim(0.4, 0.85)
    group_num += 1


plt.tight_layout()
plt.savefig('tpot-gametes-comparison.pdf', bbox_inches='tight', dpi=300)





