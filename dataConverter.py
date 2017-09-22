'''
	Inicializando classpath. 
	Depois de inicializar podem ser importados os
	demais modulos python
'''
import initContext as context
context.loadModules()

# outros modulos aqui
import logger

if __name__ == '__main__':
	log = logger.getLogger(__file__)
	log.info("testing logger")