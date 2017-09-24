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
import os
import pandas as pd

log = logger.getLogger(__file__)

def __checkDimension(X, y):
	return X.shape[0] == y.shape[0] 

def __getData(filename):
	file_exists = os.path.isfile(filename)
	if file_exists:
		log.info("Converting %s dataset" % filename)
		data = pd.read_csv(filename, header = 0)
	else:
		log.exception("file %s does not exist" % filename)
		raise IOError
	target = data['class']
	X = data.drop(['class'], axis=1)
	if not __checkDimension(X, target):
		raise ValueError("X and Y dimensions are not the same: " + str(X.shape) + " - " + str(target.shape))
	return X, target

def load_data(choice):
	log.info("Starting data conversion.")
	if choice == constants.FARKAS:
		return __getData(cdir.FARKAS_CSV)
	elif choice == constants.FEW:
		return __getData(cdir.FEW_CSV)
	elif choice == constants.ALL:
		return __getData(cdir.ALL_CSV)
	else:
		return {}

if __name__ == '__main__':
	print("Data conversion functions")
	# X, y = convertInputData(constants.FEW)