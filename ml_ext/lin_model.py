from sklearn import linear_model
from ml_ext import stats
from ml_ext import numpy
from ml_ext import pd


class LinModel(linear_model.LinearRegression):
    """
    LinearRegression class after sklearn's, but calculate t-statistics
    and p-values for model coefficients (betas).
    Additional attributes available after .fit()
    are `t` and `p` which are of the shape (y.shape[1], X.shape[1])
    which is (n_features, n_coefs)
    This class sets the intercept to 0 by default, since usually we include it
    in X.

    Original forked from https://gist.github.com/brentp/5355925
    """

    def __init__(self, *args, **kwargs):
        if not "fit_intercept" in kwargs:
            kwargs['fit_intercept'] = False
        super(LinModel, self)\
                .__init__(*args, **kwargs)
 
    def fit(self, X, y, n_jobs=1):
        self = super(LinModel, self).fit(X, y, n_jobs)

        self.nobs=X.shape[0]
        #remove an extra 1 for the alpha (k-1)
        self.df_model=X.shape[1]-1
        #(n-k-1) - we always assume an alpha is present
        self.df_resid=self.nobs-X.shape[1]-1
        #standard error of the regression 
        y_bar=y.mean()
        y_hat=self.predict(X)
        #explained sum of squares
        SSE=numpy.sum([numpy.power(y-y_bar,2) for y in y_hat])
        e=numpy.matrix(y-y_hat).T
        SST=numpy.sum([numpy.power(y-y_bar,2) for y in y])
        SSR=numpy.sum([numpy.power(x,2) for x in e])
        #print(SSR)
        
        #mean squared error of the residuals (unbiased)
        #square root of this is the standard error of the regression
        s_2 = SSR / (self.df_resid+1)
        #print(s_2)

        # #print(numpy.sqrt(numpy.diagonal(sse * numpy.linalg.inv(numpy.dot(X.T, X)))))
        # #standard error of estimator bk
        X_mat=numpy.matrix(X.values)
        #print(X_mat)
        X_dash_X=X_mat.T*X_mat
        #se = numpy.sqrt(numpy.diagonal(s_2 * numpy.linalg.inv(numpy.matrix(X.T, X))))
        se=numpy.sqrt(numpy.diagonal(s_2 * numpy.linalg.inv(X_dash_X)))

        self.se= se
        self.t = self.coef_ / se
        self.p = 2 * (1 - stats.t.cdf(numpy.abs(self.t), y.shape[0] - X.shape[1]))

        self.independent_ = []
        if isinstance(X,pd.DataFrame):
            self.independent_=X.columns.values
        #t_val=stats.t.ppf(1-0.05/2,y.shape[0] - X.shape[1])

        #print("t_val {},{}".format(t_val,self.p))
        #totla sum of squares
        
        #R2 - 1-SSR/SST
        self.rsquared=1-SSR/SST
        #adjusted r2
        #1-[(1-R2)(n-1)/(n-k-1)]
        self.rsquared_adj=1-(((1-self.rsquared)*(self.nobs-1))/self.df_resid)
        #f-value
        f_value=(self.rsquared/(self.df_model))/\
            ((1-self.rsquared)/(self.df_resid+1))
        self.f_stat=f_value
        self.f_pvalue=stats.f.pdf(f_value,self.df_model,self.df_resid+1)
        return self

    def summary(self):
        print("---------------------------------------")
        print("Variables: {}".format(self.independent_))
        print("Coefficients: {}".format(["%.4f"%x for x in self.coef_]))
        print("Std Err: {}".format(["%.4f"%x for x in self.se]))
        print("t values: {}".format(["%.4f"%x for x in self.t]))
        print("p values: {}".format(["%.4f"%x for x in self.p]))
        print("---------------------------------------")
        print("r squared: {}".format(self.rsquared))
        print("r squared adj: {}".format(self.rsquared_adj))
        print("f stat: {}".format(self.f_stat))
        print("Prob(f stat): {}".format(self.f_pvalue))
        print("N observations: {}".format(self.nobs))
        print("df_model: {}".format(self.df_model))
        print("df_resid: {}".format(self.df_resid))

    def get_confidence_interval(self,X=[]):
        """

        Calculates the confidence interval for each datapoint, given a model fit


        """
        df_results=pd.DataFrame({'y_hat'=numpy.zeros(X.shape[0])})
        y_hat=self.predict(X)
        w=numpy.matrix(df_orig[self.params.index.values].values)
        df_results['y_hat']=y_hat
        

        deg_free= self.nobs-self.params.shape[0]
        s_2=est.ssr/(deg_free)

     
        X=df_model[est.params.index.values].values
        XT_X=numpy.matrix(X).T*\
            numpy.matrix(X) 
        #print "X_XT"
        #print X_XT
        
    #    print "w"
    #    print numpy.shape(w)
    #    print "XT_T"
    #    print numpy.shape(XT_X)
        #logging.debug(numpy.shape(s_2*inv(XT_X)))
        s_c_2=numpy.array(w*s_2*inv(XT_X)*w.T)
        #logging.debug("s_c_2: {}".format(s_c_2))
        #we only want the diagonal
        s_c_2=numpy.diagonal(s_c_2)
        #logging.debug("s_c_2 diag: {}".format(s_c_2))
        #tau=df_new.apply(lambda x:numpy.matrix(x[est.params.index.values].values),axis=1)
    #        X_XT*numpy.matrix(x[est.params.index.values].values).T)
    #    tau=numpy.matrix(df_new[est.params.index.values].values[])*X_XT*\
    #        numpy.matrix(df_new[est.params.index.values].values).T
        #print "tau"
        #print numpy.shape(numpy.squeeze(tau))
        t_val=stats.t.ppf(1-alpha/2,deg_free)
        upper=E_Y+t_val*numpy.sqrt(s_c_2)
        lower=E_Y-t_val*numpy.sqrt(s_c_2)
        
        #print upper
        #print df_new.head()
        #print numpy.shape(upper)
        #print df_new.shape
    #    print "s_c_2"
    #    print numpy.shape(s_c_2)
        df_orig['s_c_2']=s_c_2
        #df_orig['sigma_tilde']=sigma_tilde
        df_orig['t']=t_val
        
        df_orig['upper_y_hat']=upper
        df_orig['lower_y_hat']=lower


