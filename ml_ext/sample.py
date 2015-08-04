"""

Calcualtes sample properties


"""
from ml_ext import statistics,numpy,stats,logging

class Sample():

	def __init__(self,X=[]):
		self.X=X

	def confidence_interval(self):
		#get sample std
		N=len(self.X)
		S=statistics.stdev(self.X)
		logging.debug(S)
		se=S/numpy.sqrt(self.X)

		#get t statistic
		pctile=stats.t.ppf(0.975,df=N-1)

		x_bar=numpy.mean(self.X)

		return (x_bar-pctile*se,x_bar+pctile*se)
