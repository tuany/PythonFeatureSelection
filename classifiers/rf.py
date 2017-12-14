#! /usr/bin/python

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
import numpy as np

def make_pipes(dimensionality_reductions):
    random_state = 1000000
    max_features = np.concatenate([["log2", "sqrt"], np.arange(0.05, 1, 0.1)])
    n_estimators = np.concatenate([np.arange(5, 100, 15), np.arange(200, 1000, 200), np.arange(1000, 11000, 5000)])
    Cs = logspace(-4, 3, 15)#concatenate([arange(0.1, 1.1, 0.1), arange(2, 6), arange(10, 60, 10)])  # all kernels
    gammas = logspace(-2, 3, 9)#(0.01, 0.1, 1.0, 10.0, 100.0)  # except linear
    one_gamma = ('auto',)
    degrees_poly = (2, 3, 4)  # only poly
    one_degree = (3,)
    # Independent term in kernel function. It is only significant in poly and sigmoid
    coef0s = (0.01, 0.1, 1.0, 10.0, 100.0)
    one_coef0 = (0.0,)
    pipes = []
    reductions_names = []
    models_names = []

    for dimensionality_reduction in dimensionality_reductions:
        for kernel in kernels:
            for C in Cs:
                gammas_variations = (gammas if kernel == 'linear' else one_gamma)
                for gamma in gammas_variations:
                    degrees_variation = (degrees_poly if kernel == 'poly' else one_degree)
                    for degree in degrees_variation:
                        coef0_variation = (coef0s if (kernel == 'poly' or kernel == 'sigmoid') else one_coef0)
                        for coef0 in coef0_variation:
                            model = SVC(kernel=kernel,
                                        C=C,
                                        gamma=gamma,
                                        degree=degree,
                                        coef0=coef0,
                                        probability=True,
                                        random_state=random_state)
                            pipe = make_pipeline(dimensionality_reduction, model)
                            pipes.append(pipe)
                            reductions_names.append(dimensionality_reduction.__class__.__name__)
                            models_names.append('svc')
    return pipes, reductions_names, models_names
