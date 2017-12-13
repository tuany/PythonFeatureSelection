#! /usr/bin/python
'''
	Inicializando classpath. 
	Depois de inicializar podem ser importados os
	demais modulos python
'''
import initContext as context
context.loadModules()

# outros modulos aqui
import logger
import classpathDir as cdir
import constants
import numpy as np
import pandas as pd
import os
from sklearn.utils import shuffle

log = logger.getLogger(__file__)
random_state=10000

def __checkDimension(X, y):
	return X.shape[0] == y.shape[0] 

def merge_frames(dataframe_list):
	return pd.concat(dataframe_list)

def remove_feature(dataframe, feature):
	return dataframe.drop(feature, axis=1)

'''
	d_type: ['euclidian', 'manhattan']
	unit: ['px', 'cm']
	dataset: ['all', 'farkas']
	m: ['1000', '']
	labels: [True, False]
'''
def load_data(d_type="euclidian", unit="px", dataset="all", m="1000", labels=False):
	log.info("Loading data from csv file")
	filename = "/distances"
	if dataset == 'all' or dataset == 'farkas':
		filename = filename + "_" + dataset

	if unit == 'px' or unit == 'cm':
		filename = filename + "_" + unit

	if m == '1000':
		filename = filename + "_" + m		

	if d_type == 'euclidian' or d_type == 'manhattan':
		filename = filename + "_" + d_type
	
	casos_file = cdir.CASES_DIR+filename+".csv"
	log.info("Casos file: %s", casos_file)
	controles_file = cdir.CONTROL_DIR_1+filename+".csv"
	log.info("Controles file: %s", controles_file)

	if os.path.isfile(casos_file) and os.path.isfile(controles_file):
		casos = pd.read_csv(casos_file, delimiter=',')
		casos_label = np.ones(len(casos), dtype=np.int)
		controles = pd.read_csv(controles_file, delimiter=',')
		controles_label = np.zeros(len(controles), dtype=np.int)

		if labels:
			casos['class'] = casos_label
			controles['class'] = controles_label

		# merge dataframes
		frames = [casos, controles]
		X = merge_frames(frames)

		# remove image paths
		X = remove_feature(X, 'img_name')
		X = remove_feature(X, 'id')
		target = np.concatenate((casos_label, controles_label))

		if not __checkDimension(X, target):
			raise ValueError("X and Y dimensions are not the same: " + str(X.shape) + " - " + str(target.shape))

		if labels:
			return shuffle(X, random_state=random_state) 
		else:
			X, target = shuffle(X, target, random_state=random_state)
			return (X, target)
	else:
		raise IOError("File not found for parameters: [" + dataset + "," + unit + "," + m + "," + d_type + "]")