from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from classifiers import svm, nb
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

	X, y = asd.load_data(d_type='manhattan', unit='px', m='', dataset='all', labels=False)
	all_samples['manhattan_px_all'] = (X, y)

	X, y = asd.load_data(d_type='euclidian', unit='px', m='', dataset='all', labels=False)
	all_samples['euclidian_px_all'] = (X, y)

	X, y = asd.load_data(d_type='manhattan', unit='px', m='1000', dataset='all', labels=False)
	all_samples['manhattan_px_1000_all'] = (X, y)

	X, y = asd.load_data(d_type='euclidian', unit='px', m='1000', dataset='all', labels=False)
	all_samples['euclidian_px_1000_all'] = (X, y)

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
		n_features_to_keep = int(0.01 * features)

		dimensionality_reductions = (
									 # PCA(n_components=n_features_to_keep),
									 # mRMRProxy(n_features_to_select=n_features_to_keep, verbose=False),
									 # FCBFProxy(n_features_to_select=n_features_to_keep, verbose=False),
									 mRMRProxy(n_features_to_select=22, verbose=False),
									 mRMRProxy(n_features_to_select=13, verbose=False),
									 mRMRProxy(n_features_to_select=8, verbose=False),
									 mRMRProxy(n_features_to_select=6, verbose=False)
									 # ReliefF(n_features_to_select=n_features_to_keep, n_neighbors=30, n_jobs=-1),
									 # RFSProxy(n_features_to_select=n_features_to_keep, verbose=False)
									 )

		pipes, reductions_names, models_names = [], [], []
		for m in [svm, nb]:
			pipe, reductions_name, models_name = m.make_pipes_lazy(dimensionality_reductions)
			pipes += pipe
			reductions_names += reductions_name
			models_names += models_name

		log.info('Total de modelos {0}'.format(len(pipes)))

		columns = ['id', 'precision', 'recall', 'f1', 'accuracy', 'dimensionality_reduction', 'error', 'classifier', 'dataset', 'n_features']

		classifiers = [SVC(), GaussianNB()]
		for classifier in classifiers:
			columns += classifier.get_params().keys()

		scoring = {'accuracy': 'accuracy',
				   'precision': 'precision',
				   'recall': 'recall',
				   'f1': 'f1'}

		with open('./output/bloco-comparacao-svm-'+k+'params-001.csv', 'wb') as csvfile:
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
					log.info("#%d - Cross-Validation success!", id)
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

					rrr = current_pipe.named_steps[reduction.lower()]
					n_features = n_features_to_keep
					if reduction != 'PCA' or reduction != 'NoneType':
						n_features = rrr.get_params(deep=False)['n_features_to_select']
					elif reduction == 'PCA':
						n_features = rrr.get_params(deep=False)['n_components']
					results = {'id': id,
							   'precision': mean_cv_results['test_precision'],
							   'recall': mean_cv_results['test_recall'],
							   'f1': mean_cv_results['test_f1'],
							   'accuracy': mean_cv_results['test_accuracy'],
							   'error': 1-mean_cv_results['test_accuracy'],
							   'dimensionality_reduction': reduction,
							   'classifier': model_name,
							   'n_features': n_features,
							   'dataset': k}
					model = current_pipe.named_steps[model_name]
					params = model.get_params(deep=False)
					log.info("#%d - Saving results!", id)
					results.update(params)
					writer.writerow(results)
				id += 1
				p_done = (100 * float(id)) / float(len(pipes))
				log.info("%.3f %% of dataset %s processing done...", p_done, k)

if __name__ == '__main__':
	start_time = time.time()
	run_combinations()
	log.info("--- Total execution time: %s minutes ---" % ((time.time() - start_time)/60))