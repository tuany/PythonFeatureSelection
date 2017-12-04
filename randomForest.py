import initContext as context
context.loadModules()
import numpy as np

from time import time
from scipy.stats import randint as sp_randint

from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import asd
import pandas as pd
import classpathDir as cd

# Utility function to report best scores
def report(results, n_top=3):
	print_results = pd.DataFrame(results)
	print_results.to_csv(cd.OUTPUT_DIR+"/rf_results_15_tree.csv", sep='\t', encoding='utf-8')
	for i in range(1, n_top + 1):
		candidates = np.flatnonzero(results['rank_test_score'] == i)
		for candidate in candidates:
			print("Model with rank: {0}".format(i))
			print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
				results['mean_test_score'][candidate],
				results['std_test_score'][candidate]))
			print("Parameters: {0}".format(results['params'][candidate]))
			print("")

# get some data
# digits = load_digits()
# X, y = digits.data, digits.target
Xdf, ydf = asd.load_data('FK')
X = Xdf.values
y = ydf.values
print("Data loaded")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=81000)

# build a classifier
clf = RandomForestClassifier(n_estimators=15)
# use a full grid over all parameters
param_grid = {
			  "max_depth": [3, 4, None],
			  "max_features": [1, 3, 4, 5, 6, 10, 15, 20],
			  "min_samples_split": [2, 3, 8, 10],
			  "min_samples_leaf": [1, 3, 5, 8, 10],
			  "bootstrap": [True, False],
			  "criterion": ["gini", "entropy"],
			  "random_state": np.random.randint(50000, 1000000, 5) }

# run grid search
grid_search = GridSearchCV(clf, param_grid=param_grid, cv=4)
start = time()
grid_search.fit(X_train, y_train)

y_pred = grid_search.predict(X_test)

cfn_matrix = confusion_matrix(y_test, y_pred)
cfn_matrix = pd.DataFrame(cfn_matrix)
cfn_matrix.to_csv(cd.OUTPUT_DIR+"/rf_results_15_tree_confusion_mtx.csv", sep='\t', encoding='utf-8')

feature_importances = grid_search.best_estimator_.feature_importances_
feature_importances = pd.DataFrame(feature_importances)
feature_importances.to_csv(cd.OUTPUT_DIR+"/rf_results_15_tree_feature_importances.csv", sep='\t', encoding='utf-8')

print("GridSearchCV took %.2f minutes for %d candidate parameter settings."
	% ((time() - start)/60, len(grid_search.cv_results_['params'])))
report(grid_search.cv_results_)