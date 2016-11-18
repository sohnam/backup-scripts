import os
import csv
import sys
import glob

# EKF + MDR
#input_txt = '/home/ansohn/Python/venvs/mdr/gametes_logs/ekf_mdr/a1000s_0.2Her/*'
#output_txt = '/home/ansohn/Python/venvs/mdr/gametes_logs/ekf_mdr/a1000s_0.2h_outputs.txt'
#input_txt = '/home/ansohn/Python/venvs/mdr/bladder_gwas_logs/b610k/ekf_mdr/*'
#output_txt = '/home/ansohn/Python/venvs/mdr/bladder_gwas_logs/b610k/b610k_ekf_mdr.txt'
#input_txt = '/home/ansohn/Python/venvs/mdr/cgems_prostate/ekf_mdr/*'
#output_txt = '/home/ansohn/Python/venvs/mdr/cgems_prostate/cgems_ekf_mdr.txt'

# EKF + genelist
#input_txt = '/home/ansohn/Python/venvs/mdr/bladder_gwas_logs/b610k/ekf_genelist/*'
#output_txt = '/home/ansohn/Python/venvs/mdr/bladder_gwas_logs/b610k/b610k_genelist.txt'

# MDR Only
#input_txt = '/home/ansohn/Python/venvs/mdr/gametes_logs/mdr_only/a5000s_0.05Her/*'
#output_txt = '/home/ansohn/Python/venvs/mdr/gametes_logs/mdr_only/a5000s_0.05h_outputs.txt'
#input_txt = '/home/ansohn/Python/venvs/mdr/bladder_gwas_logs/b610k/mdr_only/*'
#output_txt = '/home/ansohn/Python/venvs/mdr/bladder_gwas_logs/b610k/b610k_mdr_only.txt'
#input_txt = '/home/ansohn/Python/venvs/mdr/cgems_prostate/mdr_only/*'
#output_txt = '/home/ansohn/Python/venvs/mdr/cgems_prostate/cgems_mdr_only.txt'

# RF Only
#input_txt = '/home/ansohn/Python/venvs/mdr/gametes_logs/rf_only/a5000s_0.2Her/*'
#output_txt = '/home/ansohn/Python/venvs/mdr/gametes_logs/rf_only/a5000s_0.2h_outputs.txt'
#input_txt = '/home/ansohn/Python/venvs/mdr/bladder_gwas_logs/b610k/rf_only/*'
#output_txt = '/home/ansohn/Python/venvs/mdr/bladder_gwas_logs/b610k/b610k_rf_only.txt'
#input_txt = '/home/ansohn/Python/venvs/mdr/cgems_prostate/rf_only/*'
#output_txt = '/home/ansohn/Python/venvs/mdr/cgems_prostate/cgems_rf_only.txt'

# EKF Corrupt
#input_txt = '/home/ansohn/Python/venvs/mdr/gametes_logs/ekf_corrupt/a10s_0.4Her/*'
#output_txt = '/home/ansohn/Python/venvs/mdr/gametes_logs/ekf_corrupt/a10s_0.4Her/a10s_0.4_success.txt'
#input_txt = '/home/ansohn/Python/venvs/mdr/gametes_logs/ekf_corrupt/a10s_0.05Her/*'
#output_txt = '/home/ansohn/Python/venvs/mdr/gametes_logs/ekf_corrupt/a10s_0.05h_outputs.txt'

# Bladder gwas datasets
#input_dir = '/home/ansohn/Python/venvs/mdr/bladder_gwas_logs/b610k/b610k_ekf_mdr.txt'
#input_dir = '/home/ansohn/Python/venvs/mdr/bladder_gwas_logs/b610k/b610k_genelist.txt'
#input_dir = '/home/ansohn/Python/venvs/mdr/bladder_gwas_logs/b610k/b610k_mdr_only.txt'
#input_dir = '/home/ansohn/Python/venvs/mdr/bladder_gwas_logs/b610k/b610k_rf_only.txt'

#input_dir = '/home/ansohn/Python/venvs/mdr/bladder_gwas_logs/b1M/b1M_ekf_mdr.txt'
#input_dir = '/home/ansohn/Python/venvs/mdr/bladder_gwas_logs/b1M/b1M_genelist.txt'
#input_dir = '/home/ansohn/Python/venvs/mdr/bladder_gwas_logs/b1M/b1M_mdr_only.txt'
#input_dir = '/home/ansohn/Python/venvs/mdr/bladder_gwas_logs/b1M/b1M_rf_only.txt'

# EKF + MDR
#input_dir = '/home/ansohn/Python/venvs/mdr/gametes_logs/ekf_mdr/a1000s_0.2h_outputs.txt'
#input_dir = '/home/ansohn/Python/venvs/mdr/logs/ekf_mdr/a5000s.txt'
input_dir = '/home/ansohn/Python/venvs/mdr/cgems_prostate/cgems_ekf_mdr.txt'

# EKF Corrupt
#input_dir = '/home/ansohn/Python/venvs/mdr/gametes_logs/ekf_corrupt/a10s_0.05h_outputs.txt'

# MDR Only
#input_dir = '/home/ansohn/Python/venvs/mdr/gametes_logs/mdr_only/a5000s_0.05h_outputs.txt'
#input_dir = '/home/ansohn/Python/venvs/mdr/gametes_logs/mdr_only/a10s_0.1h_outputs.txt'
#input_dir = '/home/ansohn/Python/venvs/mdr/cgems_prostate/cgems_mdr_only.txt'

# RF Only
#input_dir = '/home/ansohn/Python/venvs/mdr/gametes_logs/rf_only/a5000s_0.2h_outputs.txt'
#input_dir = '/home/ansohn/Python/venvs/mdr/gametes_logs/rf_only/a10s_0.2h_outputs.txt'
#input_dir = '/home/ansohn/Python/venvs/mdr/cgems_prostate/cgems_rf_only.txt'

## Pipeline output
#input_txt = '/home/ansohn/Python/venvs/mdr/gametes_logs/ekf_mdr/a5000s_0.1Her/*'
#output_txt = '/home/ansohn/Python/venvs/mdr/gametes_logs/ekf_mdr/a5000s_0.1Her/5000_01_pipeline_output.txt'

#input_txt = '/home/ansohn/Python/venvs/mdr/cgems_prostate/ekf_mdr/*'
#output_txt = '/home/ansohn/Python/venvs/mdr/cgems_prostate/ekf_mdr/cgems_ekf_mdr_pipeline.txt'

# Before
#str_to_add = 'a1000_02h\t'
#str_to_add = 'b610k\t'
str_to_add = 'cgems_prostate\t'

# After
#str_to_add = '\tekf_mdr'
#str_to_add = '\tekf_genelist'
#str_to_add = '\tmdr_only'
#str_to_add = '\trf_only'
#str_to_add = '\tekf_corrupt'


#searchquery = 'Best pipeline'

#searchquery = '0.'
#for rf in glob.glob(input_txt):
#    with open(rf) as f1:
#        with open(output_txt, 'a') as f2:
#            lines = f1.readlines()
#            for i, line in enumerate(lines):
#                if line.startswith(searchquery):
#                    f2.write(line)


def add_str_to_lines(f_name, str_to_add):
    with open(f_name, "r") as f:
        lines = f.readlines()
        for index, line in enumerate(lines):
#             ADD BEFORE
            lines[index] = str_to_add + line.strip() + "\n"
#             ADD AFTER
#            lines[index] =  line.strip() + str_to_add + "\n"

    with open(f_name, "w") as f:
        for line in lines:
            f.write(line)
#
if __name__ == "__main__":
    add_str_to_lines(f_name=input_dir, str_to_add=str_to_add)

        
        
        
