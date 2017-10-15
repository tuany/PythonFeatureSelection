from __future__ import print_function

import numpy as np
from pyswarms.discrete import BinaryPSO
from sklearn.base import TransformerMixin

class PSOFeatureSelector(TransformerMixin):
	'''
		A feature selection algorithm using pySwarms library.
		BinaryPSO is used as wrapper feature selection. 
		Therefore, a f objective function and the classifier 
		that will evaluate the subset are needed.
	'''
	def __init__(self, alpha=0.88, classifier, dimensions, n_particles, c1, c2, w, k, p):
		self.__clf = classifier
		self.alpha = alpha
		self.__Nt = 0
		self.__Nf = 0
		self.P = 0
		self.__dimensions = dimensions
		self.__n_particles = n_particles
		self.options = {'c1': c1, 'c2': c2, 'w': w, 'k': k, 'p': p}
		self.__pso = BinaryPSO(self.__n_particles, self.__dimensions, self.options)

	'''
		Computes for the objective function per particle

		Inputs
		------
		mask : numpy.ndarray
		Binary mask that can be obtained from BinaryPSO, will
		be used to mask features.
		alpha: float 
		Constant weight for trading-off classifier performance
		and number of features

		Returns
		-------
		numpy.ndarray
		Computed objective function
	'''
	def __f_per_particle__(mask):
		# Get the subset of the features from the binary mask
		if np.count_nonzero(mask) == 0:
			X_subset = X
		else:
			X_subset = X[:,mask==1]
		# Perform classification and store performance in P
		self.__clf.fit(X_subset, y)
		self.P = (self.__clf.predict(X_subset) == y).mean()
		# Compute for the objective function
		j = (self.alpha * (1.0 - self.P) + (1.0 - self.alpha) * (1 - (X_subset.shape[1] / self.__Nf)))
		return j

	def fit(self, x, y=None):
		return self

	def transform(self, x):
		return x