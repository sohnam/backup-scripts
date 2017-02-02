import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV
from tpot.metrics import balanced_accuracy
from sklearn.metrics import make_scorer
import time


input_file = '/home/ansohn/Python/data/CGEMS-data/CGEMS-prostate-cancer-data-only-genes-predict-aggressive.csv'
data = pd.read_csv(input_file)

data = data.sample(frac=1)
features = data.drop('class', axis=1).values
labels = data['class'].values

t1 = time.time()
searchCV = LogisticRegressionCV(
        Cs=list(np.power(10.0, np.arange(-10, 10)))
        ,penalty='l2'
        ,scoring=make_scorer(balanced_accuracy)
        ,cv=10
        ,max_iter=10000
        ,solver='liblinear'
        ,n_jobs=15
)

searchCV.fit(features, labels)
t2 = time.time()

print ('Best score:', searchCV.scores_[1].mean(axis=0).max())
print ('Time elapsed:', t2 - t1)
