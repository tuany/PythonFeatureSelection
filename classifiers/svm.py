#! /usr/bin/python

from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from numpy import arange, concatenate, logspace

def make_pipes(dimensionality_reductions):
    random_state = 1000000
    kernels = ('linear', 'poly', 'rbf', 'sigmoid')
    Cs = concatenate([[0.001, 0.1, 0.3, 0.5, 0.7], [1, 5], [10, 50, 100]]) #logspace(-4, 3, 15) all kernels
    gammas = (0.001, 0.01, 0.1, 1.0, 10.0, 100.0)# logspace(-2, 3, 9)  # except linear
    one_gamma = (0.00043898156, 0.0001, 0.001, 0.003, 0.01)
    degrees_poly = (2, 3)  # only poly
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
