'''
	Main script has pipelines defined that will be used
	to compare feature selection algorithms.
	The following 4 algorithms will be compared:
	1- Particle Swarm Optimization [pyswarms]
	2- Genetic Algorithm [sklearn-deap, sklearn, deap]
	3- Correlation Feature Selection [https://github.com/shiralkarprashant/FCBF]
	4- minimum Redundance Maximum Relevance [https://pypi.python.org/pypi/mrmr/0.9.2]
	ReliefF can be used for benchmark comparison as well.

	First, the Pipeline object need to be defined considering the feature selection
	algorithms first. For the wrappers feature selection methods, the process needs 
	to be different since the objective function is the classifier itself.

	The following classifiers will be used to test the performance of the final 
	features subset: 
	1- Linear SVM (LinearSVC) or SVC
	2- RBF SVM
	3- kNN
	4- RandomForest

	The feature selection methods will be compared in terms of accuracy and precision
	of classification with the selected features subset and chosen features.	

	All these algorithms will need to be tested several times for parameter tunning 
	purposes. For PSO and GA algorithms, parameters choice is crucial.

	I've ran several tests but the results where not good (~0.5-0.7). I believe the
	low classification performance is due the parameters selection. So I will add a new
	step for parameter tunning using GridSearchCV regarding feature selection methods.
'''
import initContext as context
context.loadModules()

# outros modulos aqui
import logger
log = logger.getLogger(__file__)

import asd
import time

import pandas as pd
from sklearn.svm import SVC
from sklearn.feature_selection import RFECV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold

if __name__ == '__main__':
	# log.info("Hello world")
	# Build a classification task using 3 informative features
	# X, y = make_classification(n_samples=1000, n_features=25, n_informative=3,
	#                            n_redundant=2, n_repeated=0, n_classes=8,
	#                            n_clusters_per_class=1, random_state=0)
	start_time = time.time()
	log.info("Loading data")
	X, y = asd.load_data('FK') # all subset

	log.info("Initializing classifiers: SVM linear, SVM RBF")
	# classifiers
	global svm_linear
	svm_linear = SVC(kernel="linear")

	global svm_rbf
	svm_rbf = SVC(kernel="rbf")
	
	# split train and test data. Test subset is 1/3 of set
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1000)
	log.debug("Train set size: " + str(X_train.shape) + "\tTest set size: " + str(X_test.shape))

	log.info("Fitting SVM Linear")
	svm_linear.fit(X_train, y_train)
	log.info("Done!")
	log.info("Fitting SVM RBF")
	svm_rbf.fit(X_train, y_train)
	log.info("Done!")

	svm_linear_p = svm_linear.score(X_test, y_test)
	svm_rbf_p = svm_rbf.score(X_test, y_test)

	# The "accuracy" scoring is proportional to the number of correct
	# classifications
	rfecv1 = RFECV(estimator=svm_linear, step=1, cv=StratifiedKFold(10),
	              scoring='accuracy')
	rfecv1.fit(X, y)
	selected_features = rfecv1.get_support(True)
	log.info("Optimal number of features : %d" % rfecv1.n_features_)
	log.info("Selected features indexes: " + str(selected_features))

	X_train = pd.DataFrame(X_train).iloc[:, selected_features].values
	X_test = pd.DataFrame(X_test).iloc[:, selected_features].values
	log.info("Fitting SVM Linear after feature selection")

	svm_linear.fit(X_train, y_train)
	log.info("Done!")
	svm_linear_q = svm_linear.score(X_test, y_test)

	svm_rbf.fit(X_train, y_train)
	log.info("Done!")
	svm_rbf_q = svm_rbf.score(X_test, y_test)

	print('SVM Linear performance bf feature selection: %.3f' % (svm_linear_p))
	print('SVM RBF performance bf feature selection: %.3f' % (svm_rbf_p))
	print('SVM Linear performance af feature selection: %.3f' % (svm_linear_q))
	print('SVM RBF performance af feature selection: %.3f' % (svm_rbf_q))
	log.info("--- Total execution time: %s minutes ---" % ((time.time() - start_time)/60))