from sklearn import svm
import time as t
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler
import asd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import classpathDir as cd
import itertools

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

random_state = 854789
start_time = t.time()
params = {
			'kernel': ['linear', 'rbf', 'poly', 'sigmoid'], 
			'C': [1, 5, 10, 50, 100, 500,1000], # np.logspace(-4, 3, 15)
			'gamma': [0.05,0.1, 0.5, 1, 10, 50, 100],  # np.logspace(-4, 3, 15)
			'degree': [2, 3], 
			'random_state': np.random.randint(50000, 1000000, 3) }#[42]}

Xdf, ydf = asd.load_data('FW')
X = Xdf.values
y = ydf.values
print("Data loaded")
# PSO selected features [0,1,2,3,4,5,6,7]
X = X[:, 0:8]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=random_state)
# X_train = X
# X_test = X
# y_train = y
# y_test = y

linear_param = {'C': params['C'], 'gamma': params['gamma'], 'random_state': params['random_state']}
linear_svc = svm.SVC(kernel='linear')
clf = GridSearchCV(linear_svc, linear_param, cv=StratifiedKFold(n_splits=3), scoring=['accuracy', 'precision_macro', 'recall_macro'], refit='accuracy')
print("Executing grid search")
clf.fit(X_train, y_train)

best_params = clf.best_params_
print("Linear kernel SVM best params:")
print(clf.best_params_)
print("Best score based on accuracy: %.6f" % clf.best_score_)
results = pd.DataFrame(clf.cv_results_)
results.to_csv(cd.OUTPUT_DIR+"/gs_results_linear_svm_accuracy.csv", sep='\t', encoding='utf-8')

y_pred = clf.predict(X_test)

cfn_matrix = confusion_matrix(y_test, y_pred)
cfn_matrix = pd.DataFrame(cfn_matrix)
cfn_matrix.to_csv(cd.OUTPUT_DIR+"/gs_results_linear_svm_cm.csv", sep='\t', encoding='utf-8')

np.set_printoptions(precision=2)
# Plot normalized confusion matrix
class_names=['ASD', 'NON-ASD']
plt.figure()
plot_confusion_matrix(cfn_matrix.values, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')

# plt.show()
# # plt.scatter(X[y==1][1], X[y==1][2], label='ASD', c='red', marker='o')
# # plt.scatter(X[y==0][1], X[y==0][2], label='NORMAL', c='blue', marker='^')

# # # Prettify the graph
# # plt.legend()
# # plt.xlabel("Feature 1")
# # plt.ylabel("Feature 2")

# # # display
# # plt.show()

# '''
# 	To plot the n-dimensional problem in a 2D graph we need to choose
# 	only 2 features.
# 	So I will choose the 2 most discriminant features
# '''
# # print("Reshaping")
# # X = Xdf.values[:, 1:3] # we only take the first two features. We could
# #                        # avoid this ugly slicing by using a two-dim dataset
# # print(X)
# # print("Reshape ok")
# # clf = svm.SVC(kernel='linear', C=best_params['C'], 
# # 				gamma=best_params['gamma'], random_state=best_params['random_state'])
# # clf.fit(X, y)

# # plt.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=plt.cm.Paired)

# # # plot the decision function
# # ax = plt.gca()
# # xlim = ax.get_xlim()
# # ylim = ax.get_ylim()

# # # create grid to evaluate model
# # xx = np.linspace(xlim[0], xlim[1], 30)
# # yy = np.linspace(ylim[0], ylim[1], 30)
# # YY, XX = np.meshgrid(yy, xx)
# # xy = np.vstack([XX.ravel(), YY.ravel()]).T
# # Z = clf.decision_function(xy).reshape(XX.shape)

# # # plot decision boundary and margins
# # ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
# #            linestyles=['--', '-', '--'])
# # # plot support vectors
# # ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=100,
# #            linewidth=1, facecolors='none')
# # plt.show()


rbf_param = { 'C': params['C'],
			  'gamma': params['gamma'],
			  'random_state': params['random_state']
			}

rbf_svc = svm.SVC(kernel='rbf')
clf = GridSearchCV(rbf_svc, rbf_param, cv=StratifiedKFold(n_splits=3), scoring=['accuracy', 'precision_macro', 'recall_macro'], refit='accuracy')
print("Executing grid search")
clf.fit(X_train, y_train)

best_params = clf.best_params_
print("RBF kernel SVM best params:")
print(clf.best_params_)
print("Best score based on accuracy: %.6f" % clf.best_score_)
results = pd.DataFrame(clf.cv_results_)
results.to_csv(cd.OUTPUT_DIR+"/gs_results_rbf_svm_accuracy.csv", sep='\t', encoding='utf-8')

y_pred = clf.predict(X_test)

cfn_matrix = confusion_matrix(y_test, y_pred)
cfn_matrix = pd.DataFrame(cfn_matrix)
cfn_matrix.to_csv(cd.OUTPUT_DIR+"/gs_results_rbf_svm_cm.csv", sep='\t', encoding='utf-8')

np.set_printoptions(precision=2)
# Plot normalized confusion matrix
class_names=['ASD', 'NON-ASD']
plt.figure()
plot_confusion_matrix(cfn_matrix.values, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')

poly_param = { 'C': params['C'],
			         'gamma': params['gamma'],
			         'degree': params['degree'],
			         'random_state': params['random_state']
			       }
poly_svc = svm.SVC(kernel='poly')
clf = GridSearchCV(poly_svc, poly_param, cv=StratifiedKFold(n_splits=3), scoring=['accuracy', 'precision_macro', 'recall_macro'], refit='accuracy')
clf.fit(X_train, y_train)

best_params = clf.best_params_
print("Poly kernel SVM best params:")
print(clf.best_params_)
print("Best score based on accuracy: %.6f" % clf.best_score_)
results = pd.DataFrame(clf.cv_results_)
results.to_csv(cd.OUTPUT_DIR+"/gs_results_poly_svm_accuracy.csv", sep='\t', encoding='utf-8')

y_pred = clf.predict(X_test)

cfn_matrix = confusion_matrix(y_test, y_pred)
cfn_matrix = pd.DataFrame(cfn_matrix)
cfn_matrix.to_csv(cd.OUTPUT_DIR+"/gs_results_poly_svm_cm.csv", sep='\t', encoding='utf-8')

np.set_printoptions(precision=2)
# Plot normalized confusion matrix
class_names=['ASD', 'NON-ASD']
plt.figure()
plot_confusion_matrix(cfn_matrix.values, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')

sigmoid_param = { 'C': params['C'],
                  'gamma': params['gamma'],
                  'degree': params['degree'],
                  'random_state': params['random_state']
                }
sigmoid_svc = svm.SVC(kernel='sigmoid')
clf = GridSearchCV(sigmoid_svc, sigmoid_param, cv=StratifiedKFold(n_splits=3), scoring=['accuracy', 'precision_macro', 'recall_macro'], refit='accuracy')
clf.fit(X_train, y_train)
best_params = clf.best_params_
print("Sigmoid kernel SVM best params:")
print(clf.best_params_)
print("Best score based on accuracy: %.6f" % clf.best_score_)
results = pd.DataFrame(clf.cv_results_)
results.to_csv(cd.OUTPUT_DIR+"/gs_results_sigmoid_svm_accuracy.csv", sep='\t', encoding='utf-8')

y_pred = clf.predict(X_test)

cfn_matrix = confusion_matrix(y_test, y_pred)
cfn_matrix = pd.DataFrame(cfn_matrix)
cfn_matrix.to_csv(cd.OUTPUT_DIR+"/gs_results_sigmoid_svm_cm.csv", sep='\t', encoding='utf-8')

np.set_printoptions(precision=2)
# Plot normalized confusion matrix
class_names=['ASD', 'NON-ASD']
plt.figure()
plot_confusion_matrix(cfn_matrix.values, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')

print("GridSearchCV Finished at %s minutes" % str((t.time()-start_time)/60) )