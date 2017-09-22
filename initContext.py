import sys
import os

def loadModules():
	sys.path.insert(0, os.getcwd()+"/config")

if __name__ == '__main__':
	loadModules()