import os


directoryname = "/home/ansohn/Python/venvs/mdr/gametes_logs/mdr_only/a5000s_0.05Her"

tpot_files = os.listdir(directoryname)
print(tpot_files)

for i in range(len(tpot_files)):
    os.rename(
        os.path.join(directoryname, tpot_files[i]),
        os.path.join(directoryname, str(i+1)+'.log')
        )