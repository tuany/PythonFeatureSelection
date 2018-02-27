import numpy as np


def code_test_samples(samples):
    dim = len(samples.shape)
    if dim > 1:
        return np.concatenate((samples[0:10, 0:10], samples[-11:-1, -11:-1]), axis=0)
    else:
        return np.concatenate((samples[0:10], samples[-11:-1]), axis=0)


def mean_scores(scores):
    mean_score = {}
    for score_key, score_value in scores.items():
        mean_score[score_key] = np.mean(score_value, axis=0)
    return mean_score
