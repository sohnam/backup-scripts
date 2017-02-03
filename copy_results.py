import os
import csv
import sys
import glob
import mmap


# EKF Only
#input_txt = '/home/ansohn/Python/Results/gametes/logit/a10/h04/*.out'
#output_txt = '/home/ansohn/Python/Results/gametes/logit/a10/h04/logit_a10_04_results.txt'
#output_txt = '/home/ansohn/Python/Results/gametes/ekf_mdr/a10/h02/ekfmdr_a10_02_pipeline.txt'

input_txt = '/home/ansohn/Python/Results/cgems/logit/logit_cgems_results.txt'
output_txt = '/home/ansohn/Python/Results/cgems/logit/logit_cgems_results_2.txt'


#searchquery = 'Best pipeline'
#searchquery = 'XGB_cgems'
#searchquery = 'Best score'
#searchquery = 'Generation 200'
#searchquery = '0.'

#for rf in glob.glob(input_txt):
#    with open(rf) as f1:
#        with open(output_txt, 'a') as f2:
#            lines = f1.readlines()
#            for i, line in enumerate(lines):
#                if line.startswith(searchquery):
#                    f2.write(line)


f = open(input_txt,'r')
filedata = f.read()
f.close()

newdata = filedata.replace("logit", "Logistic Regression")

f = open(output_txt,'w')
f.write(newdata)
f.close()


# Before
#str_to_add = 'a10_02h\t'
#str_to_add = 'cgems_prostate\t'

## After
#str_to_add = '\tekf_mdr'
#str_to_add = '\tmdr_only'
#str_to_add = '\txgb_only'
#str_to_add = '\tmdr_pred'
#str_to_add = '\tlogit'


#def add_str_to_lines(f_name, str_to_add):
#    with open(f_name, "r") as f:
#        lines = f.readlines()
#        for index, line in enumerate(lines):
##             ADD BEFORE
##            lines[index] = str_to_add + line.strip() + "\n"
##             ADD AFTER
#            lines[index] =  line.strip() + str_to_add + "\n"
#
#    with open(f_name, "w") as f:
#        for line in lines:
#            f.write(line)
##
#if __name__ == "__main__":
#    add_str_to_lines(f_name=input_txt, str_to_add=str_to_add)  
