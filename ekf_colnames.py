# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 13:16:43 2016

@author: ansohn
"""
#import os
#import fnmatch
import glob


#idx = 0
#for name in glob.glob('/home/ansohn/Python/loc2/relieff*.txt'):
#    idx = idx + glob.glob('/home/ansohn/Python/loc2/relieff*.txt').count(name)
#print(idx)

for name in glob.glob('/home/ansohn/Python/data/Genomics_data/bladder-scores/*.txt'):
#for name in glob.glob('/home/ansohn/sarlacc_data/bladder-gwas/bladder-scores/*.txt'):        
    if "=== SCORES ===\nGene\tScore\tRank" in open(name).read():
        text = open(name).read()
        open(name, "w").write(text.replace("=== SCORES ===\nGene\tScore\tRank", "=== SCORES ==="))
    else:
        pass
    
    if "=== SCORES ===\nGene\tRank\tScore" in open(name).read():
        text = open(name).read()
        open(name, "w").write(text.replace("=== SCORES ===\nGene\tRank\tScore", "=== SCORES ===\nGene\tScore\tRank"))
    else:
        pass
    
    if "=== SCORES ===" in open(name).read():
        text = open(name).read()
        open(name, "w").write(text.replace("=== SCORES ===", "=== SCORES ===\nGene\tScore\tRank"))
    

#for name in glob.glob('/home/ansohn/Python/venvs/mdr/logs/ekf_mdr/a5000s_0.1h_outputs.txt'):
#    appendfile = open(name, "a")
#    appendfile.write("Feature_combo\tScore")

#def find_files(directory, pattern):
#    if not os.path.exists(directory):
#        raise ValueError("Directory not found {}".format(directory))
#
#    matches = []
#    for root, dirnames, filenames in os.walk(directory):
#        for filename in filenames:
#            full_path = os.path.join(root, filename)
#            if fnmatch.filter([full_path], pattern):
#                matches.append(os.path.join(root, filename))
#    return matches




    
