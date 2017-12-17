#! /usr/bin/python

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
import numpy as np

def make_pipes(dimensionality_reductions):
    random_state = 1000000
    max_features = np.concatenate([["log2", "sqrt"], np.arange(0.05, 1, 0.1)])
    n_estimators = np.concatenate([np.arange(5, 100, 15), np.arange(200, 1000, 200), np.arange(1000, 11000, 5000)])
    min_sample_leafs = np.concatenate([np.arange(0.05, 0.3, 0.05)], [100, 200])
    max_depths = (30, 50, 100, 230)

    pipes = []
    reductions_names = []
    models_names = []

    dimensionality_reduction = None

    for max_feature in max_features:
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
                                oob_score=True,
                                n_jobs=-1,
                                random_state=random_state,
                                class_weight=None)
                    pipe = make_pipeline(dimensionality_reduction, model)
                    pipes.append(pipe)
                    reductions_names.append(dimensionality_reduction.__class__.__name__)
                    models_names.append('randomForest')
    return pipes, reductions_names, models_names
