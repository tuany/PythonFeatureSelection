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
import csv

log = logger.getLogger(__file__)

def __getData(filename):
	data = {}
	file = open(filename, 'rb')
	reader = csv.reader(file)
	headers = reader.next()
	for row in reader:
		for h, v in zip(headers, row):
			if h == "class":
				data[h.strip()] = int(v)
			else:	
				data[h.strip()] = float(v)
	
	file.close()
	return data

def __few_set():
	few_dict = {}
	filename = cdir.FEW_CSV
	file_exists = os.path.isfile(filename)
	if file_exists:
		log.info("Converting few dataset")
		few_dict = __getData(filename)
		log.info("Features for few dataset acquired!")
		log.info(few_dict.keys()[0])
	else:
		log.exception("file %s does not exist" % filename)
		raise IOError
	return few_dict

def __farkas_set():
	farkas_dict = {}
	filename = cdir.FARKAS_CSV
	file_exists = os.path.isfile(filename)
	if file_exists:
		log.info("Converting farkas dataset")
		farkas_dict = __getData(filename)
		log.info("Features for farkas dataset acquired!")
		log.info(farkas_dict.keys()[0])
	else:
		log.exception("file %s does not exist" % filename)
		raise IOError	
	return farkas_dict

def __all_set():
	all_dict = {}

	filename = cdir.ALL_CSV
	file_exists = os.path.isfile(filename)
	if file_exists:
		log.info("Converting all dataset")
		all_dict = __getData(filename)
		log.info("Features for all dataset acquired!")
		log.info(all_dict.keys()[0])
	else:
		log.exception("file %s does not exist" % filename)
		raise IOError

	return all_dict

def convertInputData(choice):
	log.info("Starting data conversion.")
	if choice == constants.FARKAS:
		return __farkas_set()
	elif choice == constants.FEW:
		return __few_set()
	elif choice == constants.ALL:
		return __all_set()
	else:
		return {}

if __name__ == '__main__':
	print("Data conversion functions")