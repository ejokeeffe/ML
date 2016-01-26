from sklearn import linear_model
from ml_ext import stats
from ml_ext import numpy
from ml_ext import pd
from ml_ext import logging
from ml_ext import inv
from ml_ext import plt
from ml_ext import sns
from sklearn import metrics as skmetrics

def get_MAPE(y=[],y_hat=[]):
	"""

	Get the mean absolute percentage error
	$MAPE=100\frac{1}{n}\sum_{i=0}^{n-1}\left | \frac{y_i-\hat{y_i}}{y_i} \right |$

	"""
	n=len(y)
	MAPE=0
	for ii in range(n):
		MAPE=MAPE+numpy.abs((y[ii]-y_hat[ii])/y[ii])
	MAPE=100*MAPE/n
	return MAPE

def get_RMSE_pc(y=[],y_hat=[]):
	"""

	Get the mean absolute percentage error
	$RMSE_pc\frac{100}{\bar{y}}\sqrt{\frac{1}{n}\sum_{i=0}^{n-1} ( y_i-\hat{y_i} )^2}$

	"""
	
	rmse=numpy.sqrt(skmetrics.mean_squared_error(y,y_hat))
	rmse_pc=100*rmse/numpy.mean(y)
	return rmse_pc