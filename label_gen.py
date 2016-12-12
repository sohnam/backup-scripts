# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 14:06:26 2016

@author: ansohn
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb
import pandas as pd
#from glob import glob
#from collections import defaultdict

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

# MDR with P0/P1
mdr_perfect = pd.read_csv('/home/ansohn/Python/venvs/mdr/gametes_logs/target_scores.txt', sep='\t')

experiment_data = pd.concat([ekf_all, mdr_all, rf_all, mdr_perfect])

dataset = ['a10_005h', 'a10_01h', 'a10_02h', 'a10_04h', 
           'a100_005h', 'a100_01h', 'a100_02h', 'a100_04h', 
           'a1000_005h', 'a1000_01h', 'a1000_02h', 'a1000_04h',
           'a5000_005h', 'a5000_01h', 'a5000_02h', 'a5000_04h']

plt.figure(figsize=(14, 0.75))
plt.ylim(0, 10)
for i in range(4):
    plt.barh([0], [0], color=sb.color_palette('Paired')[i], #color=sb.light_palette('black')[i],
             label=['TPOT (MDR only)', 'Random Forest', 'TPOT (MDR + EKF)', 'MDR (Predicative SNPs)'][i])

plt.legend(fontsize=14, ncol=4)
plt.savefig('tpot-gametes-comparison-legend.pdf', bbox_inches='tight', dpi=300)
