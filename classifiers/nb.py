from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline


def make_pipes(dimensionality_reductions):
    pipes = []
    reductions_names = []
    models_names = []

    for dimensionality_reduction in dimensionality_reductions:
        model = GaussianNB()
        pipe = make_pipeline(dimensionality_reduction, model)
        pipes.append(pipe)
        reductions_names.append(dimensionality_reduction.__class__.__name__)
        models_names.append('gaussiannb')
    return pipes, reductions_names, models_names

def make_pipes_lazy(dimensionality_reductions):
    return make_pipes(dimensionality_reductions)