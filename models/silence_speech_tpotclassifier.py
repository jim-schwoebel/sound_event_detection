import numpy as np 
import json, pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# NOTE: Make sure that the class is labeled 'target' in the data file
g=json.load(open('silence_speech_tpotclassifier_.json'))
tpot_data=g['labels']
features=g['data']

training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data, random_state=None)

# Average CV score on the training set was:1.0
exported_pipeline = RandomForestClassifier(bootstrap=False, criterion="entropy", max_features=0.45, min_samples_leaf=4, min_samples_split=9, n_estimators=100)

exported_pipeline.fit(training_features, training_target)
print('saving classifier to disk')
f=open('silence_speech_tpotclassifier.pickle','wb')
pickle.dump(exported_pipeline,f)
f.close()
