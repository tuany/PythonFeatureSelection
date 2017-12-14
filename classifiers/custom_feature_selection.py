from __future__ import print_function
from skfeature.function.information_theoretical_based import MRMR, FCBF
from skfeature.function.statistical_based import CFS
from skfeature.function.sparse_learning_based import RFS
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import inspect

class mRMR(BaseEstimator, TransformerMixin):
	def __init__(self, n_features_to_select=2, mode='rank', verbose=True):
		self.n_features_to_select = n_features_to_select
		self.mode = mode
		self.verbose = verbose

	def fit(self, X, y):
		self._X = X
		self._y = y
		self.ranking_ = MRMR.mrmr(self._X, self._y, self.mode)
		if self.verbose:
			print("Feature ranking: " + str(self.ranking_))
		return self

	def transform(self, X):
		return X[:, self.ranking_[:self.n_features_to_select]]

	def fit_transform(self, X, y):
		self.fit(X, y)
		return self.transform(X)

class FCBF(BaseEstimator, TransformerMixin):
	def __init__(self, n_features_to_select=2, mode='rank', delta=0.0, verbose=True):
		self.n_features_to_select = n_features_to_select
		self.delta = delta
		self.mode = mode
		self.verbose = verbose

	def fit(self, X, y):
		self._X = X
		self._y = y
		self.ranking_ = FCBF.fcbf(self._X, self._y, kwargs={'delta':self.delta})
		if self.verbose:
			print("Feature ranking: " + str(self.ranking_))
		return self

	def transform(self, X):
		return X[:, self.ranking_[:self.n_features_to_select]]

	def fit_transform(self, X, y):
		self.fit(X, y)
		return self.transform(X)

class CFS(BaseEstimator, TransformerMixin):
	def __init__(self, n_features_to_select=None, mode='rank', verbose=True):
		self.n_features_to_select = n_features_to_select
		self.mode = mode
		self.verbose = verbose

	def fit(self, X, y):
		self._X = X
		self._y = y
		self.ranking_ = CFS.cfs(self._X, self._y, self.mode)
		if self.verbose:
			print("Feature ranking: " + str(self.ranking_))
		return self

	def transform(self, X):
		if self.n_features_to_select == None:
			return X[:, self.ranking_]

		return X[:, self.ranking_[:self.n_features_to_select]]

	def fit_transform(self, X, y):
		self.fit(X, y)
		return self.transform(X)

class RFS(BaseEstimator, TransformerMixin):
	def __init__(self, n_features_to_select=None, mode='rank', gamma=1, verbose=True):
		self.n_features_to_select = n_features_to_select
		self.mode = mode
		self.verbose = verbose
		self.gamma=gamma

	def fit(self, X, y):
		self._X = X
		self._y = y
		self.ranking_ = RFS.rfs(self._X, self._y, self.mode, kwargs={'gamma':self.gamma, 'verbose':self.verbose})
		if self.verbose:
			print("Feature ranking: " + str(self.ranking_))
		return self

	def transform(self, X):
		if self.n_features_to_select == None:
			return X[:, self.ranking_]

		return X[:, self.ranking_[:self.n_features_to_select]]

	def fit_transform(self, X, y):
		self.fit(X, y)
		return self.transform(X)