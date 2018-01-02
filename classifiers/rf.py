#! /usr/bin/python

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
import numpy as np
from numpy.random import randint

def make_pipes(dimensionality_reductions):
    random_states = randint(100000, 999999, 10)
    max_features = ("log2", "sqrt")
    n_estimators = (500, 1000)#np.concatenate([np.arange(1000, 11000, 5000), np.arange(100, 1000, 500)])
    min_sample_leafs = np.concatenate([np.arange(0.01, 0.06, 0.00325)])
    max_depths = (50, 120)

    pipes = []
    reductions_names = []
    models_names = []

    dimensionality_reduction = None
    for max_feature in max_features:
        for random_state in random_states:
            for n_estimator in n_estimators:
                for min_sample_leaf in min_sample_leafs:
                    for max_depth in max_depths:
                        model = RandomForestClassifier(
                                    n_estimators=n_estimator,
                                    criterion='gini',
                                    max_depth=max_depth,
                                    min_samples_leaf=min_sample_leaf,
                                    max_features=max_feature,
                                    bootstrap=True,
                                    n_jobs=-1,
                                    random_state=random_state,
                                    class_weight=None)
                        pipe = make_pipeline(dimensionality_reduction, model)
                        pipes.append(pipe)
                        reductions_names.append(dimensionality_reduction.__class__.__name__)
                        models_names.append('randomforestclassifier')
    return pipes, reductions_names, models_names
