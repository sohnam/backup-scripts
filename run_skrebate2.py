import pandas as pd
import skrebate
import numpy as np

############## Her 0.1 ###################
input_file = '/home/ansohn/Python/data/gametes-data/loc2_filtered/a100/Her01/a_100s_2000her_0.1__maf_0.2_EDM-1_01.txt'
rel_out = '/home/ansohn/Python/data/gametes-data/a100_h01_relieff.txt'
surf_out = '/home/ansohn/Python/data/gametes-data/a100_h01_surf.txt'
surfstar_out = '/home/ansohn/Python/data/gametes-data/a100_h01_surfstar.txt'
msurf_out = '/home/ansohn/Python/data/gametes-data/a100_h01_msurf.txt'

data = pd.read_csv(input_file, sep='\t')
labels = data['Class'].values
features = data.drop('Class', axis=1)

rel = skrebate.ReliefF()
surf = skrebate.SURF()
surfstar = skrebate.SURFstar()
msurf = skrebate.MultiSURF()

#rel1 = np.savetxt(rel_out, rel.fit(features.values, labels).top_features_.astype('int32'))
#surf1 = np.savetxt(surf_out, surf.fit(features.values, labels).top_features_.astype('int32'))
#surfstar1 = np.savetxt(surfstar_out, surfstar.fit(features.values, labels).top_features_.astype('int32'))
#msurf1 = np.savetxt(msurf_out, msurf.fit(features.values, labels).top_features_.astype('int32'))



############## Her 0.02 ###################
#input_file = '/home/ansohn/Python/data/gametes-data/loc2_filtered/a100/Her02/a_100s_2000her_0.2__maf_0.2_EDM-1_01.txt'
#rel_out = '/home/ansohn/Python/data/gametes-data/a100_h02_relieff.txt'
#surf_out = '/home/ansohn/Python/data/gametes-data/a100_h02_surf.txt'
#surfstar_out = '/home/ansohn/Python/data/gametes-data/a100_h02_surfstar.txt'
#msurf_out = '/home/ansohn/Python/data/gametes-data/a100_h02_msurf.txt'
#
#data = pd.read_csv(input_file, sep='\t')
#labels = data['Class'].values
#features = data.drop('Class', axis=1)
#
#rel = skrebate.ReliefF()
#surf = skrebate.SURF()
#surfstar = skrebate.SURFstar()
#msurf = skrebate.MultiSURF()
#
#rel1 = np.savetxt(rel_out, rel.fit(features.values, labels).top_features_.astype('int32'))
#surf1 = np.savetxt(surf_out, surf.fit(features.values, labels).top_features_.astype('int32'))
#surfstar1 = np.savetxt(surfstar_out, surfstar.fit(features.values, labels).top_features_.astype('int32'))
#msurf1 = np.savetxt(msurf_out, msurf.fit(features.values, labels).top_features_.astype('int32'))



############## Her 0.04 ###################
#input_file = '/home/ansohn/Python/data/gametes-data/loc2_filtered/a100/Her04/a_100s_2000her_0.4__maf_0.2_EDM-1_01.txt'
#rel_out = '/home/ansohn/Python/data/gametes-data/a100_h04_relieff.txt'
#surf_out = '/home/ansohn/Python/data/gametes-data/a100_h04_surf.txt'
#surfstar_out = '/home/ansohn/Python/data/gametes-data/a100_h04_surfstar.txt'
#msurf_out = '/home/ansohn/Python/data/gametes-data/a100_h04_msurf.txt'
#
#data = pd.read_csv(input_file, sep='\t')
#labels = data['Class'].values
#features = data.drop('Class', axis=1)
#
#rel = skrebate.ReliefF()
#surf = skrebate.SURF()
#surfstar = skrebate.SURFstar()
#msurf = skrebate.MultiSURF()
#
#rel1 = np.savetxt(rel_out, rel.fit(features.values, labels).top_features_.astype('int32'))
#surf1 = np.savetxt(surf_out, surf.fit(features.values, labels).top_features_.astype('int32'))
#surfstar1 = np.savetxt(surfstar_out, surfstar.fit(features.values, labels).top_features_.astype('int32'))
#msurf1 = np.savetxt(msurf_out, msurf.fit(features.values, labels).top_features_.astype('int32'))



############## Her 0.05 ###################
#input_file = '/home/ansohn/Python/data/gametes-data/loc2_filtered/a100/Her005/a_100s_2000her_0.05__maf_0.2_EDM-1_01.txt'
#rel_out = '/home/ansohn/Python/data/gametes-data/a100_h005_relieff.txt'
#surf_out = '/home/ansohn/Python/data/gametes-data/a100_h005_surf.txt'
#surfstar_out = '/home/ansohn/Python/data/gametes-data/a100_h005_surfstar.txt'
#msurf_out = '/home/ansohn/Python/data/gametes-data/a100_h005_msurf.txt'
#
#data = pd.read_csv(input_file, sep='\t')
#labels = data['Class'].values
#features = data.drop('Class', axis=1)
#
#rel = skrebate.ReliefF()
#surf = skrebate.SURF()
#surfstar = skrebate.SURFstar()
#msurf = skrebate.MultiSURF()
#
#rel1 = np.savetxt(rel_out, rel.fit(features.values, labels).top_features_.astype('int32'))
#surf1 = np.savetxt(surf_out, surf.fit(features.values, labels).top_features_.astype('int32'))
#surfstar1 = np.savetxt(surfstar_out, surfstar.fit(features.values, labels).top_features_.astype('int32'))
#msurf1 = np.savetxt(msurf_out, msurf.fit(features.values, labels).top_features_.astype('int32'))