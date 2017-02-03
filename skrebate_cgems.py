import pandas as pd
import skrebate
import numpy as np


############## Her 0.1 ###################
input_file = '/home/ansohn/data/CGEMS-data/CGEMS-prostate-cancer-data-only-genes-predict-aggressive.csv'

#rel_out = '/home/ansohn/cgems/cgems_relieff.txt'
#surf_out = '/home/ansohn/cgems/cgems_surf.txt'
#surfstar_out = '/home/ansohn/cgems/cgems_surfstar.txt'
msurf_out = '/home/ansohn/cgems/cgems_msurf.txt'

data = pd.read_csv(input_file)
labels = data['class'].values
features = data.drop('class', axis=1)

#rel = skrebate.ReliefF(n_features_to_select=2, n_jobs=-1)
#surf = skrebate.SURF(n_features_to_select=2, n_jobs=-1)
#surfstar = skrebate.SURFstar(n_features_to_select=2, n_jobs=-1)
msurf = skrebate.MultiSURF(n_features_to_select=2, n_jobs=-1)

#rel1 = np.savetxt(rel_out, rel.fit(features.values, labels).top_features_.astype('int32'))
#surf1 = np.savetxt(surf_out, surf.fit(features.values, labels).top_features_.astype('int32'))
#surfstar1 = np.savetxt(surfstar_out, surfstar.fit(features.values, labels).top_features_.astype('int32'))
msurf1 = np.savetxt(msurf_out, msurf.fit(features.values, labels).top_features_.astype('int32'))
