"""

Provides examples of how to use the library

"""
from ml_ext import pd
from ml_ext import logging
from ml_ext import random

random.seed(10)
def sample_properties():
	"""

	Pop out some sample properties for explanation

	"""

	#read in some wooldridge data
	df=pd.read_csv('http://samba.fsv.cuni.cz/~cahlik/Backup/Ekonometrie/Data%20Wooldridge%20Stata/401k.csv')

	logging.debug(df.describe())

	#sample mean
