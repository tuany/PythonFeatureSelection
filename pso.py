'''
	Particle Swarm Optimization algorithm as wrapper feature selection.
	For BinaryPSO the position of particle is defined as:
		x = [x1, x2, x3, ..., xd] where xi => [0,1]
		d is the number of features.

	PSO needs a objective function, as a wrapper feature selection method, the
	objective function is the classifier.

	Since, I've been using sklearn Pipeline, it is needed to define this class
	as ModelTransform and pass the classifier as parameter.
'''

# Import modules
import numpy as np
import seaborn as sns
import pandas as pd
import asd 
import time

# Import PySwarms
import pyswarms as ps
from sklearn.datasets import make_classification
# from sklearn import linear_model
from sklearn.svm import SVC, LinearSVC

# Define objective function
def f_per_particle(m, alpha):
    """Computes for the objective function per particle

    Inputs
    ------
    m : numpy.ndarray
        Binary mask that can be obtained from BinaryPSO, will
        be used to mask features.
    alpha: float (default is 0.5)
        Constant weight for trading-off classifier performance
        and number of features

    Returns
    -------
    numpy.ndarray
        Computed objective function
    """
    total_features = 15
    # Get the subset of the features from the binary mask
    if np.count_nonzero(m) == 0:
        X_subset = X
    else:
        X_subset = X[:,m==1]
    # Perform classification and store performance in P
    classifier.fit(X_subset, y)
    P = (classifier.predict(X_subset) == y).mean()
    # Compute for the objective function
    j = (alpha * (1.0 - P)
        + (1.0 - alpha) * (1 - (X_subset.shape[1] / total_features)))

    return j

def f(x, alpha=0.88):
	"""Higher-level method to do classification in the
	whole swarm.

	Inputs
	------
	x: numpy.ndarray of shape (n_particles, dimensions)
	    The swarm that will perform the search

	Returns
	-------
	numpy.ndarray of shape (n_particles, )
	    The computed loss for each particle
	"""
	n_particles = x.shape[0]
	j = [f_per_particle(x[i], alpha) for i in range(n_particles)]
	return np.array(j)


start_time = time.time()
# X, y = make_classification(n_samples=100, n_features=15, n_classes=3, 
# 							n_informative=4, n_redundant=1, n_repeated=2, random_state=1)
X, y = asd.load_data('FK') # farkas subset
print(X.shape)
print(y.shape)

X = X.values
y = y.values

df = pd.DataFrame(X)
df['labels'] = pd.Series(y)

# Create an instance of the classifier
global classifier
# classifier = linear_model.LogisticRegression()
classifier = SVC(kernel="rbf")
# classifier = LinearSVC()

# Initialize swarm, arbitrary
# c1 = cognitive parameter, float
# c2 =  social parameter, float
# w = inertia parameter, float
# k = number of neighbors to be considered, int and k < n_particles
# p = {1,2}, Minkowski p-norm. 1: sum-of-absolute values (L1), 2: euclidean distance (L2) 
options = {'c1': 1.4, 'c2': 1.5, 'w':0.7, 'k': 20, 'p':2}

# Call instance of PSO
dimensions = X.shape[1] # dimensions should be the number of features
# optimizer.reset()
optimizer = ps.discrete.BinaryPSO(n_particles=60, dimensions=dimensions, options=options)

# Perform optimization
cost, pos = optimizer.optimize(f, print_step=10, iters=400, verbose=2)


# Get the selected features from the final positions
X_selected_features = X[:,pos==1]  # subset

# Perform classification and store performance in P
classifier.fit(X_selected_features, y)

# Compute performance
subset_performance = (classifier.predict(X_selected_features) == y).mean()


print('Subset performance: %.3f' % (subset_performance))
print("--- Total execution time: %s minutes ---" % ((time.time() - start_time)/60))