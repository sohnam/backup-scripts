from collections import Counter
import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

gametes_filename = '/home/ansohn/Python/data/tpot_mdr_gametes_data/a-100/Her-0.2/a_100s_2000her_0.2__maf_0.2_EDM-1_01.txt'
load_gametes = pd.read_csv(gametes_filename, sep='\t')

phenotype = load_gametes['Class']
individuals = load_gametes.drop('Class', axis=1)

X_train, X_test, y_train, y_test = train_test_split(individuals, phenotype, 
                                                    train_size=0.75, 
                                                    test_size=0.25, 
                                                    random_state=42)

training_data = pd.DataFrame(X_train)
training_data['class'] = y_train
training_data['group'] = 'training'

testing_data = pd.DataFrame(X_test)
testing_data['class'] = 0
testing_data['group'] = 'testing'

training_testing_data = pd.concat([training_data, testing_data])
most_frequent_class = Counter(X_train).most_common(1)[0][0]
training_testing_data['guess'] = most_frequent_class

non_feature_columns = ['class', 'group', 'guess']

training_features = training_testing_data.loc[training_testing_data['group'] == 'training'].drop(non_feature_columns, axis=1)
training_classes = training_testing_data.loc[training_testing_data['group'] == 'training', 'class'].values

testing_features = training_testing_data.loc[training_testing_data['group'] == 'testing'].drop(non_feature_columns, axis=1)
testing_classes = training_testing_data.loc[training_testing_data['group'] == 'testing', 'class'].values

clf = RandomForestClassifier(max_depth=5, max_features=len(training_features.columns), 
                             n_estimators=1000, random_state=1)
clf.fit(training_features, training_classes)
print(clf.score(testing_features, testing_classes))

#parameters = {'max_depth':[1,2,3,4,5], 
#              'max_features':[1,2,3,4,5,10,20,30,40,50,60,70,80,90,100], 
#              'n_estimators':[500, 1000]}
#rf = RandomForestClassifier()
#clf = GridSearchCV(estimator=rf, param_grid=parameters, n_jobs=8)
#clf.fit(training_features, training_classes)
#print(clf.best_params_)



