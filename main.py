from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from classifiers import svm, nb
from classifiers.custom_feature_selection import mRMR, FCBF, CFS, RFS
from sklearn.model_selection import StratifiedKFold, cross_validate
import asd
import csv
import progressbar
from sklearn.decomposition import PCA
from skrebate import ReliefF
from classifiers.utils import code_test_samples, mean_scores
import time
import initContext as context
context.loadModules()
# outros modulos aqui
import logger
log = logger.getLogger(__file__)

def run_combinations():
	# X, y = asd.load_data(d_type='euclidian', unit='px', m='1000', dataset='all', labels=False)
	# X, y = asd.load_data(d_type='manhattan', unit='px', m='1000', dataset='all', labels=False)
	# X, y = asd.load_data(d_type='euclidian', unit='px', m='', dataset='all', labels=False)
	# X, y = asd.load_data(d_type='manhattan', unit='px', m='', dataset='all', labels=False)

	# X, y = asd.load_data(d_type='euclidian', unit='px', m='1000', dataset='farkas', labels=False)
	# X, y = asd.load_data(d_type='manhattan', unit='px', m='1000', dataset='farkas', labels=False)
	# X, y = asd.load_data(d_type='euclidian', unit='px', m='', dataset='farkas', labels=False)
	samples, labels = asd.load_data(d_type='manhattan', unit='px', m='', dataset='farkas', labels=False)
	samples = samples.values
	n_samples = samples.shape[0]
	n_features = samples.shape[1]

	# less samples for fast coding test
	# samples = code_test_samples(samples.values)
	# labels = code_test_samples(labels)
	instances, features = samples.shape
	print('Data has {0} instances and {1} features'.format(instances, features))
	n_features_to_keep = int(0.1 * features)

	dimensionality_reductions = (None,
								 PCA(n_components=n_features_to_keep),
								 ReliefF(n_features_to_select=n_features_to_keep, n_neighbors=10, n_jobs=1),
								 mRMR(n_features_to_select=n_features_to_keep, verbose=False),
								 FCBF(n_features_to_select=n_features_to_keep, verbose=False),
								 CFS(n_features_to_select=n_features_to_keep, verbose=False),
								 RFS(n_features_to_select=n_features_to_keep, verbose=False)
								 )

	pipes, reductions_names, models_names = [], [], []
	for m in [svm, nb]:
		pipe, reductions_name, models_name = m.make_pipes(dimensionality_reductions)
		pipes += pipe
		reductions_names += reductions_name
		models_names += models_name

	print('Total de modelos {0}'.format(len(pipes)))

	columns = ['id', 'precision', 'recall', 'f1', 'accuracy', 'dimensionality_reduction', 'error', 'classifier']

	classifiers = [SVC(), GaussianNB()]
	for classifier in classifiers:
		columns += classifier.get_params().keys()

	scoring = {'accuracy': 'accuracy',
			   'precision': 'precision',
			   'recall': 'recall',
			   'f1': 'f1'}

	with open('./output/classifiers.csv', 'wb') as csvfile:
		writer = csv.DictWriter(csvfile, fieldnames=columns)
		writer.writeheader()
		id = 0
		for current_pipe, reduction, model_name in zip(pipes, reductions_names, models_names):
			try:
				cv = StratifiedKFold(n_splits=4)
				cv_results = cross_validate(estimator=current_pipe,
											X=samples,
											y=labels,
											scoring=scoring,
											cv=cv,
											n_jobs=1)  # all CPUs
			except ValueError as e:
				print(e)
				cv_results = None
			except KeyError as ke:
				print(ke)
				cv_results = None

			if cv_results is not None:
				mean_cv_results = mean_scores(cv_results)
				results = {'id': id,
						   'precision': mean_cv_results['test_precision'],
						   'recall': mean_cv_results['test_recall'],
						   'f1': mean_cv_results['test_f1'],
						   'accuracy': mean_cv_results['test_accuracy'],
						   'error': 1-mean_cv_results['test_accuracy'],
						   'dimensionality_reduction': reduction,
						   'classifier': model_name}
				model = current_pipe.named_steps[model_name]
				params = model.get_params(deep=False)
				results.update(params)
				writer.writerow(results)
			id += 1

if __name__ == '__main__':
	start_time = time.time()
	run_combinations()
	log.info("--- Total execution time: %s minutes ---" % ((time.time() - start_time)/60))