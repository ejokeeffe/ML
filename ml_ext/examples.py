"""

Provides examples of how to use the library

"""
from ml_ext import pd
from ml_ext import logging
from ml_ext import random
from ml_ext import numpy

random.seed(10)

def sample_properties():
	"""

	Pop out some sample properties for explanation

	"""

	#read in some wooldridge data
	df=pd.read_csv('http://samba.fsv.cuni.cz/~cahlik/Backup/Ekonometrie/Data%20Wooldridge%20Stata/401k.csv')

	logging.debug(df.describe())

	#sample mean

def gen_simplemodel_data(n=1000,k=1):
	numpy.random.seed(10)
	df_x=pd.DataFrame({'alpha':numpy.ones(n)})
	coefs=numpy.random.rand(k+1)
	for ii in range(k):
		#draw from normal distribution and scale it randomly
		df_x['X{}'.format(ii)]=numpy.random.normal(0,1,n)

	# logging.debug(df_x.head())
	# logging.debug("coefs: {}. df_x: {}. disturb: {}".format(\
	# 	numpy.shape(numpy.matrix(coefs)),\
	# 	numpy.shape(numpy.matrix(df_x)),\
	# 	numpy.shape(numpy.matrix(numpy.random.normal(0,1,n)*numpy.random.rand(1)).T)))
	data=numpy.array(numpy.matrix(df_x)*numpy.matrix(coefs).T+numpy.matrix(numpy.random.normal(0,1,n)*numpy.random.rand(1)).T)
	df_x['y']=data

	return (coefs,df_x)
