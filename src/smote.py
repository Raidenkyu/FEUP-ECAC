# Oversample with SMOTE and random undersample for imbalanced dataset
from collections import Counter
from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from matplotlib import pyplot
from numpy import where

def smote_sampling(x_train, y_train):
  
    # summarize class distribution
    counter = Counter(y_train)
    print(counter)
    # define pipeline
    over = SMOTE(sampling_strategy=0.2)
    under = RandomUnderSampler(sampling_strategy=0.5)
    steps = [('o', over), ('u', under)]
    pipeline = Pipeline(steps=steps)
    # transform the dataset
    X_result, y_result = pipeline.fit_resample(x_train, y_train)
    # summarize the new class distribution
    counter = Counter(y_result)
    print(counter)
   

    return X_result, y_result