#! /usr/bin/python

from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from numpy import arange, concatenate, logspace
from numpy.random import randint

def make_pipes(dimensionality_reductions):
    random_state = 4353452
    kernels = ('linear', 'poly', 'rbf', 'sigmoid')
    Cs = concatenate([[0.3], [5], [10]]) #logspace(-4, 3, 15) all kernels
    gammas = (0.01, 0.1, 10.0)# logspace(-2, 3, 9)  # except linear
    one_gamma = (0.0001, 0.01)
    degrees_poly = (2, 3)  # only poly
    one_degree = (3,)
    # Independent term in kernel function. It is only significant in poly and sigmoid
    coef0s = (1.0, 100.0)
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

def make_pipes_lazy(dimensionality_reductions):
    pipes = []
    reductions_names = []
    models_names = []
    model1 = SVC(kernel='linear',
                C=5,
                gamma=1,
                degree=3,
                coef0=0.0,
                probability=True,
                random_state=1000000)

    model2 = SVC(kernel='poly',
                C=0.3,
                gamma=0.001,
                degree=2,
                coef0=0.01,
                probability=True,
                random_state=1000000)

    model3 = SVC(kernel='rbf',
                C=0.7,
                gamma=0.0001,
                degree=3,
                coef0=0.0,
                probability=True,
                random_state=1000000)

    model4 = SVC(kernel='sigmoid',
                C=0.001,
                gamma=0.001,
                degree=3,
                coef0=1,
                probability=True,
                random_state=734628)


    models = [model1, model2, model3, model4]
    for dimensionality_reduction in dimensionality_reductions:
        for m in models:
            pipe = make_pipeline(dimensionality_reduction, m)
            pipes.append(pipe)
            reductions_names.append(dimensionality_reduction.__class__.__name__)
            models_names.append('svc')
    return pipes, reductions_names, models_names