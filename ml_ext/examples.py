"""

Provides examples of how to use the library

"""
from ml_ext import pd
from ml_ext import logging
from ml_ext import random
from ml_ext import numpy
from ml_ext import lin_model
from ml_ext import plt
from ml_ext import sns

random.seed(10)
nupmy.random.seed(10)

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

def run_simple_regression(n=1000,k=1,feature='X0'):
	(coefs,df)=gen_simplemodel_data(n=n,k=k)
	# logging.debug(df.head())
	df.sort(feature,inplace=True)
	lr=lin_model.LinModel()
	X=df[df.columns[df.columns!='y']]
	y=df.y


	lr.fit(X=X,y=y)
	lr.summary()
	df_ci=lr.get_confidence_interval_for_mean(X)
	df_pi=lr.get_prediction_interval(X)


	(fig,ax)=plt.subplots(nrows=2,ncols=1,figsize=[14,12])

	cols=sns.color_palette('husl',n_colors=4)
	ax[0].scatter(X[feature],y,label='y',color=cols[3],alpha=0.4)

	ax[0].plot(X[feature],df_pi['upper_pred'],label='pred',color=cols[1],alpha=0.5)
	ax[0].plot(X[feature],df_pi['lower_pred'],color=cols[1],alpha=0.5)
	ax[0].plot(X[feature],df_ci['upper_mean'],color=cols[2],alpha=0.5)
	ax[0].plot(X[feature],df_ci['lower_mean'],label='mean_ci',color=cols[2],alpha=0.5)
	ax[0].scatter(X[feature],df_pi['y_hat'],label='y_hat',color=cols[0],alpha=0.5)
	ax[0].legend(loc='best')

	ax[1].scatter(X[feature],y,label='y',color=cols[3],alpha=0.4)
	ax[1].scatter(X[feature],df_ci['y_hat'],label='y_hat',color=cols[0],alpha=0.5)
	ax[1].legend(loc='best')

