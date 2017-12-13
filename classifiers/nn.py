#! /usr/bin/python

from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from numpy import arange, concatenate


def make_pipes(dimensionality_reductions):
    hidden_layer_sizes = concatenate([arange(0.1, 1.1, 0.1), arange(10, 110, 10)])
    learning_rate = 'adaptive'
    activations = ('identity', 'logistic', 'tanh', 'relu')
    # momentums = (0.1, 0.5, 0.9)
    max_iters = (10000000,)

    pipes = []
    reductions_names = []
    models_names = []

    for dimensionality_reduction in dimensionality_reductions:
        for hidden_layer_size in hidden_layer_sizes:
            for activation in activations:
                # for momentum in momentums:
                    for max_iter in max_iters:
                        model = MLPClassifier(hidden_layer_sizes=hidden_layer_size,
                                              learning_rate=learning_rate,
                                              activation=activation,
                                              # momentum=momentum,
                                              max_iter=max_iter)
                        pipe = make_pipeline(dimensionality_reduction, model)
                        pipes.append(pipe)
                        reductions_names.append(dimensionality_reduction.__class__.__name__)
                        models_names.append('mlpclassifier')
    return pipes, reductions_names, models_names
