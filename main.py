from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from classifiers import svm, nb, nn
from classifiers.custom_feature_selection import mRMRProxy, FCBFProxy, CFSProxy, RFSProxy
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
	all_samples = {}

	X, y = asd.load_data(d_type='euclidian', unit='px', m='1000', dataset='all', labels=False)
	all_samples['euclidian_px_1000_all'] = (X, y)

	X, y = asd.load_data(d_type='manhattan', unit='px', m='1000', dataset='all', labels=False)
	all_samples['manhattan_px_1000_all'] = (X, y)

	X, y = asd.load_data(d_type='euclidian', unit='px', m='', dataset='all', labels=False)
	all_samples['euclidian_px_all'] = (X, y)

	X, y = asd.load_data(d_type='manhattan', unit='px', m='', dataset='all', labels=False)
	all_samples['manhattan_px_all'] = (X, y)

	X, y = asd.load_data(d_type='euclidian', unit='px', m='1000', dataset='farkas', labels=False)
	all_samples['euclidian_px_1000_farkas'] = (X, y)

	X, y = asd.load_data(d_type='manhattan', unit='px', m='1000', dataset='farkas', labels=False)
	all_samples['manhattan_px_1000_farkas'] = (X, y)

	X, y = asd.load_data(d_type='euclidian', unit='px', m='', dataset='farkas', labels=False)
	all_samples['euclidian_px_farkas'] = (X, y)

	X, y = asd.load_data(d_type='manhattan', unit='px', m='', dataset='farkas', labels=False)
	all_samples['manhattan_px_farkas'] = (X, y)

	for k in all_samples.keys():
		log.info("Running models for %s dataset", k)
		samples, labels = all_samples[k]
		log.info("X.shape %s, y.shape %s", str(samples.shape), str(labels.shape))
		samples = samples.values
		n_samples = samples.shape[0]
		n_features = samples.shape[1]

		# less samples for fast coding test
		# samples = code_test_samples(samples.values)
		# labels = code_test_samples(labels)
		instances, features = samples.shape
		log.info('Data has {0} instances and {1} features'.format(instances, features))
		n_features_to_keep = int(0.1 * features)

		dimensionality_reductions = (None,
									 PCA(n_components=n_features_to_keep),
									 ReliefF(n_features_to_select=n_features_to_keep, n_neighbors=10, n_jobs=-1),
									 mRMRProxy(n_features_to_select=n_features_to_keep, verbose=False),
									 FCBFProxy(n_features_to_select=n_features_to_keep, verbose=False),
									 CFSProxy(n_features_to_select=n_features_to_keep, verbose=False),
									 RFSProxy(n_features_to_select=n_features_to_keep, verbose=False)
									 )

		pipes, reductions_names, models_names = [], [], []
		for m in [svm, nb, nn]:
			pipe, reductions_name, models_name = m.make_pipes(dimensionality_reductions)
			pipes += pipe
			reductions_names += reductions_name
			models_names += models_name

		log.info('Total de modelos {0}'.format(len(pipes)))

		columns = ['id', 'precision', 'recall', 'f1', 'accuracy', 'dimensionality_reduction', 'error', 'classifier', 'dataset']

		classifiers = [SVC(), GaussianNB(), MLPClassifier()]
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
					log.info("Executing Cross-Validation")
					cv = StratifiedKFold(n_splits=4)
					cv_results = cross_validate(estimator=current_pipe,
												X=samples,
												y=labels,
												scoring=scoring,
												cv=cv,
												n_jobs=-1)  # all CPUs
					log.info("Cross-Validation success!")
				except ValueError as e:
					log.exception("Exception during pipeline execution", extra=e)
					cv_results = None
				except KeyError as ke:
					log.exception("Exception during pipeline execution", extra=ke)
					cv_results = None

				if cv_results is not None:
					mean_cv_results = mean_scores(cv_results)
					log.info("#%d - CV result (accuracy) %.2f for model %s and reduction %s", 
							id, mean_cv_results['test_accuracy'], model_name, reduction)
					results = {'id': id,
							   'precision': mean_cv_results['test_precision'],
							   'recall': mean_cv_results['test_recall'],
							   'f1': mean_cv_results['test_f1'],
							   'accuracy': mean_cv_results['test_accuracy'],
							   'error': 1-mean_cv_results['test_accuracy'],
							   'dimensionality_reduction': reduction,
							   'classifier': model_name,
							   'dataset': k}
					model = current_pipe.named_steps[model_name]
					params = model.get_params(deep=False)
					log.info("#%d - Saving results!", id)
					results.update(params)
					writer.writerow(results)
				id += 1

if __name__ == '__main__':
	start_time = time.time()
	run_combinations()
	log.info("--- Total execution time: %s minutes ---" % ((time.time() - start_time)/60))