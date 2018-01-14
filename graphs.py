from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from classifiers.custom_feature_selection import mRMRProxy, FCBFProxy, CFSProxy, RFSProxy
from sklearn.model_selection import StratifiedKFold, cross_validate
import asd
import csv
from sklearn.decomposition import PCA
from skrebate import ReliefF
from classifiers.utils import code_test_samples, mean_scores
import time
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
import initContext as context
context.loadModules()
# outros modulos aqui
import logger
log = logger.getLogger(__file__)

all_samples = {}

X, y = asd.load_data(d_type='manhattan', unit='px', m='', dataset='all', labels=False)
all_samples['manhattan_px_all'] = (X, y)

# X, y = asd.load_data(d_type='euclidian', unit='px', m='', dataset='all', labels=False)
# all_samples['euclidian_px_all'] = (X, y)

# X, y = asd.load_data(d_type='manhattan', unit='px', m='1000', dataset='all', labels=False)
# all_samples['manhattan_px_1000_all'] = (X, y)

# X, y = asd.load_data(d_type='euclidian', unit='px', m='1000', dataset='all', labels=False)
# all_samples['euclidian_px_1000_all'] = (X, y)

n_features_to_keep=13
feature_selection = CFSProxy(n_features_to_select=n_features_to_keep, verbose=False)
new_X = feature_selection.fit_transform(X.values, y)
r = feature_selection.ranking_

clf = SVC(kernel='linear',
	        C=5,
	        gamma=1,
	        degree=3,
	        coef0=0.0,
	        probability=True,
	        random_state=1000000)

clf.fit(new_X, y)