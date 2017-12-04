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
from scipy import interp
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import warnings
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc
from sklearn.feature_selection import RFECV, SelectKBest, f_classif, mutual_info_classif
from sklearn.model_selection import train_test_split, StratifiedKFold

def versiontuple(v):
    return tuple(map(int, (v.split("."))))


def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):

    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, c=cmap(idx),
                    marker=markers[idx], label=cl)

    # highlight test samples
    if test_idx:
        # plot all samples
        if not versiontuple(np.__version__) >= versiontuple('1.9.0'):
            X_test, y_test = X[list(test_idx), :], y[list(test_idx)]
            warnings.warn('Please update to NumPy 1.9.0 or newer')
        else:
            X_test, y_test = X[test_idx, :], y[test_idx]

        plt.scatter(X_test[:, 0],
                    X_test[:, 1],
                    c='',
                    alpha=1.0,
                    linewidths=1,
                    marker='o',
                    s=55, label='test set')

if __name__ == '__main__':
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
	# {u'C': 316.227766017, u'random_state': 933561, u'gamma': 0.0001}
	svm_linear = SVC(kernel="linear", C=316.227766017, gamma=0.0001, random_state=933561,  probability=True)

	global svm_rbf
	# {u'C': 1.0, u'random_state': 933561, u'gamma': 10.0}
	svm_rbf = SVC(kernel="rbf", C=1.0, gamma=10.0, random_state=933561)

	global svm_poly
	# {u'C': 0.0001, u'random_state': 933561, u'gamma': 316.227766017, u'degree': 2}
	svm_poly = SVC(kernel="poly", C=0.0001, gamma=316.227766017, degree=2, random_state=933561)

	global svm_sigmoid
	# {u'C': 316.227766017, u'random_state': 933561, u'gamma': 0.0316227766017, u'degree': 2}
	svm_sigmoid = SVC(kernel="sigmoid", C=316.227766017, gamma=0.0316227766017, degree=2, random_state=933561)	

	clfs = [svm_linear] #, svm_rbf, svm_poly, svm_sigmoid]
	# split train and test data. Test subset is 1/3 of set
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1000)
	log.debug("Train set size: " + str(X_train.shape) + "\tTest set size: " + str(X_test.shape))
	log.info("Fitting SVM Linear")
	svm_linear.fit(X_train, y_train)
	log.info("Done!")
	svm_linear_q = svm_linear.score(X_test, y_test)

	# univariate = SelectKBest(mutual_info_classif, k=4)
	# univariate.fit(X_train, y_train)
	# selected_features = univariate.get_support(True)
	# log.info("Optimal number of features univariate : %d" % len(selected_features))
	# log.info("Selected features indexes: univariate" + str(selected_features))

	rfecv1 = RFECV(estimator=svm_linear, step=1, cv=StratifiedKFold(4),
	              scoring='accuracy')
	rfecv1.fit(X.values, y.values)
	selected_features = rfecv1.get_support(True)
	log.info("Optimal number of features RFECV : %d" % rfecv1.n_features_)
	log.info("Selected features indexes: RFECV" + str(selected_features))
	X_train = pd.DataFrame(X_train).iloc[:, selected_features].values
	X_test = pd.DataFrame(X_test).iloc[:, selected_features].values

	log.info("Fitting SVM Linear after feature selection")
	svm_linear.fit(X_train, y_train)
	log.info("Done!")
	svm_linear_q = svm_linear.score(X_test, y_test)
	print('SVM Linear performance af feature selection: %.3f' % (svm_linear_q))

	A = X.iloc[:, selected_features].values
	a = y.values

	for clf in clfs:
		cv = StratifiedKFold(n_splits = 4)
		tprs = []
		aucs = []
		mean_fpr = np.linspace(0, 1, 100)
		i = 0
		for train, test in cv.split(A, a):
		    probas_ = clf.fit(A[train], a[train]).predict_proba(A[test])
		    # Compute ROC curve and area the curve
		    fpr, tpr, thresholds = roc_curve(a[test], probas_[:, 1])
		    tprs.append(interp(mean_fpr, fpr, tpr))
		    tprs[-1][0] = 0.0
		    roc_auc = auc(fpr, tpr)
		    aucs.append(roc_auc)
		    plt.plot(fpr, tpr, lw=1, alpha=0.3,
		             label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))

		    i += 1
		plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
		         label='Luck', alpha=.8)
		mean_tpr = np.mean(tprs, axis=0)
		mean_tpr[-1] = 1.0
		mean_auc = auc(mean_fpr, mean_tpr)
		std_auc = np.std(aucs)
		plt.plot(mean_fpr, mean_tpr, color='b',
		         label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
		         lw=2, alpha=.8)

		std_tpr = np.std(tprs, axis=0)
		tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
		tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
		plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
		                 label=r'$\pm$ 1 std. dev.')

		plt.xlim([-0.05, 1.05])
		plt.ylim([-0.05, 1.05])
		plt.xlabel('False Positive Rate')
		plt.ylabel('True Positive Rate')
		plt.title('Receiver operating characteristic example')
		plt.legend(loc="lower right")
		plt.show()

	# log.info("Fitting SVM RBF")
	# svm_rbf.fit(X_train, y_train)
	# log.info("Done!")
	# log.info("Fitting SVM Poly")
	# svm_poly.fit(X_train, y_train)
	# log.info("Done!")
	# log.info("Fitting SVM Sigmoid")
	# svm_sigmoid.fit(X_train, y_train)
	# log.info("Done!")

	# svm_linear_p = svm_linear.score(X_test, y_test)
	# svm_rbf_p = svm_rbf.score(X_test, y_test)
	# svm_poly_p = svm_poly.score(X_test, y_test)
	# svm_sigmoid_p = svm_sigmoid.score(X_test, y_test)

	# log.info("Fitting SVM RBF after feature selection")
	# svm_rbf.fit(X_train, y_train)
	# log.info("Done!")
	# svm_rbf_q = svm_rbf.score(X_test, y_test)

	# log.info("Fitting SVM Poly after feature selection")
	# svm_poly.fit(X_train, y_train)
	# log.info("Done!")
	# svm_poly_q = svm_poly.score(X_test, y_test)

	# log.info("Fitting SVM Sigmoid after feature selection")
	# svm_sigmoid.fit(X_train, y_train)
	# log.info("Done!")
	# svm_sigmoid_q = svm_sigmoid.score(X_test, y_test)

	# print('SVM Linear performance bf feature selection: %.3f' % (svm_linear_p))
	# print('SVM RBF performance bf feature selection: %.3f' % (svm_rbf_p))
	# print('SVM Poly performance bf feature selection: %.3f' % (svm_poly_p))
	# print('SVM Sigmoid performance bf feature selection: %.3f' % (svm_sigmoid_p))

	# print('SVM Linear performance af feature selection: %.3f' % (svm_linear_q))
	# print('SVM RBF performance af feature selection: %.3f' % (svm_rbf_q))
	# print('SVM Poly performance af feature selection: %.3f' % (svm_poly_q))
	# print('SVM Sigmoid performance af feature selection: %.3f' % (svm_sigmoid_q))



	# plt.show()
# # plt.scatter(X[y==1][1], X[y==1][2], label='ASD', c='red', marker='o')
# # plt.scatter(X[y==0][1], X[y==0][2], label='NORMAL', c='blue', marker='^')

# # # Prettify the graph
# # plt.legend()
# # plt.xlabel("Feature 1")
# # plt.ylabel("Feature 2")

# # # display
# # plt.show()

	'''
		To plot the n-dimensional problem in a 2D graph we need to choose
		only 2 features.
		So I will choose the 2 most discriminant features
	'''
	# X_graph = X.iloc[:, selected_features].values # we only take the first two features. We could
	#                        # avoid this ugly slicing by using a two-dim dataset
	# plt.scatter(X_graph[:, 0], X_graph[:, 1], c=y, s=30, cmap=plt.cm.Paired)

	# # plot the decision function
	# ax = plt.gca()
	# xlim = ax.get_xlim()
	# ylim = ax.get_ylim()

	# # create grid to evaluate model
	# xx = np.linspace(xlim[0], xlim[1], 30)
	# yy = np.linspace(ylim[0], ylim[1], 30)
	# YY, XX = np.meshgrid(yy, xx)
	# xy = np.vstack([XX.ravel(), YY.ravel()]).T
	# Z = svm_linear.decision_function(xy).reshape(XX.shape)

	# # plot decision boundary and margins
	# ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
	#            linestyles=['--', '-', '--'])
	# # plot support vectors
	# ax.scatter(svm_linear.support_vectors_[:, 0], svm_linear.support_vectors_[:, 1], s=100,
	#            linewidth=1, facecolors='none')
	# plt.show()

	# plot_decision_regions(X_graph, y, classifier=svm_linear)
	# plt.legend(loc='upper left')
	# plt.tight_layout()
	# plt.show()

	# plot_decision_regions(X_graph, y, classifier=svm_rbf)
	# plt.legend(loc='upper left')
	# plt.tight_layout()
	# plt.show()

	# plot_decision_regions(X_graph, y, classifier=svm_poly)
	# plt.legend(loc='upper left')
	# plt.tight_layout()
	# plt.show()

	# plot_decision_regions(X_graph, y, classifier=svm_sigmoid)
	# plt.legend(loc='upper left')
	# plt.tight_layout()
	# plt.show()
	
	log.info("--- Total execution time: %s minutes ---" % ((time.time() - start_time)/60))