from sklearn import linear_model
from ml_ext import stats
from ml_ext import numpy
from ml_ext import pd
from ml_ext import logging
from ml_ext import inv
from ml_ext import plt
from ml_ext import sns


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
        """
        y can be series or array
        X can be dataframe or ndarry (N datapoints x M features)
        """
        self = super(LinModel, self).fit(X, y, n_jobs)


        self.nobs=X.shape[0]
        self.nparams=X.shape[1]
        #remove an extra 1 for the alpha (k-1)
        self.df_model=X.shape[1]-1
        #(n-k-1) - we always assume an alpha is present
        self.df_resid=self.nobs-X.shape[1]-1
        #standard error of the regression 
        y_bar=y.mean()
        y_hat=self.predict(X)
        self.fittedvalues=y_hat
        #explained sum of squares
        SSE=numpy.sum([numpy.power(val-y_bar,2) for val in y_hat])
        e=numpy.matrix(y-y_hat).T
        self.resid=numpy.ravel(e)
        # logging.debug(y_bar)
        # logging.debug(y)
        SST=numpy.sum([numpy.power(val-y_bar,2) for val in y])
        SSR=numpy.sum([numpy.power(x,2) for x in e])
        #print(SSR)
        
        #mean squared error of the residuals (unbiased)
        #square root of this is the standard error of the regression
        s_2 = SSR / (self.df_resid+1)
        self.s_y=numpy.sqrt(s_2)
        #print(s_2)

        #Also get the means of the independent variables
        if isinstance(X,pd.core.frame.DataFrame):
            #assume its' called alpha
            self.X_bar=X[X.columns[X.columns!='alpha']].mean()
            Z=numpy.matrix(X[X.columns[X.columns!='alpha']])
        else:
            #assume its the first column
            self.X_bar=numpy.mean(X.values,axis=0)[1:]
            Z=numpy.matrix(X[:,1:])
        
        i_n=numpy.matrix(numpy.ones(self.nobs))
        M_0=numpy.matrix(numpy.eye(self.nobs))-numpy.power(self.nobs,-1)*i_n*i_n.T
        self.Z_M_Z=Z.T*M_0*Z
        # #print(numpy.sqrt(numpy.diagonal(sse * numpy.linalg.inv(numpy.dot(X.T, X)))))
        # #standard error of estimator bk
        X_mat=numpy.matrix(X.values)
        #print(X_mat)
        self.X_dash_X=X_mat.T*X_mat
        # we get nans using this approach so calculate each one separately
        # se=numpy.zeros(self.nparams)
        # for ii in range(self.nparams):
        #     se[ii]=numpy.sqrt(X_dash_X[ii,ii]*s_2)
        # logging.debug(s_2)
        # logging.debug(numpy.linalg.inv(X_dash_X))
        # #se = numpy.sqrt(numpy.diagonal(s_2 * numpy.linalg.inv(numpy.matrix(X.T, X))))
        se=numpy.sqrt(numpy.diagonal(s_2 * numpy.linalg.inv(self.X_dash_X)))

        self.se= se
        self.t = self.coef_ / se
        self.p = 2 * (1 - stats.t.cdf(numpy.abs(self.t), y.shape[0] - X.shape[1]))

        self.independent_ = []
        if isinstance(X,pd.DataFrame):
            self.independent_=X.columns.values
        #t_val=stats.t.ppf(1-0.05/2,y.shape[0] - X.shape[1])

        
        
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
        # return self

    def get_confidence_intervals_for_coefs(self):
        """

        Gets the upper and lower bound 95% confidence intervals for
        each coefficient excluding alpha

        t_k=(b_k-beta_k)/sqrt(s^2 * Skk)

        """
        alpha=0.05
        t_val=stats.t.ppf(1-alpha/2,self.df_resid+1)

        df_res=pd.DataFrame({'upper':numpy.zeros(len(self.coef_)),\
            'lower':numpy.zeros(len(self.coef_)),\
            'b':self.coef_},index=self.independent_)

        for ii,var in enumerate(self.independent_):
            df_res.loc[var,'upper']=df_res.loc[var,'b']+t_val*self.se[ii]
            df_res.loc[var,'lower']=df_res.loc[var,'b']-t_val*self.se[ii]

        return df_res

    def get_coefs_as_ts(self):
        return pd.Series(self.coef_,index=self.independent_)

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

    def get_confidence_interval_for_mean(self,X=[]):
        """

        Calculates the confidence interval for each datapoint, given a model fit
        This is the confidence interval of the model, not the prediction interval


        """
        if isinstance(X,pd.core.frame.DataFrame):
            X=X[self.independent_]
        df_results=pd.DataFrame({'y_hat':numpy.zeros(X.shape[0])})
        y_hat=self.predict(X)
        w=numpy.matrix(X)

     
        # XT_X=numpy.matrix(X).T*\
        #     numpy.matrix(X) 
        #print "X_XT"
        #print X_XT
        
    #    print "w"
    #    print numpy.shape(w)
    #    print "XT_T"
    #    print numpy.shape(XT_X)
        #logging.debug(numpy.shape(s_2*inv(XT_X)))
        s_c_2=numpy.array(w*numpy.power(self.s_y,2)*inv(self.X_dash_X)*w.T)
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
        #95% confidence interval so alpha =0.95
        alpha=0.05
        t_val=stats.t.ppf(1-alpha/2,self.df_resid+1)
        upper=y_hat+t_val*numpy.sqrt(s_c_2)
        lower=y_hat-t_val*numpy.sqrt(s_c_2)
        

        # df_orig['s_c_2']=s_c_2
        # #df_orig['sigma_tilde']=sigma_tilde
        # df_orig['t']=t_val
        
        # df_orig['upper_y_hat']=upper
        # df_orig['lower_y_hat']=lower
        df=pd.DataFrame({'y_hat':y_hat,'upper_mean':upper,'lower_mean':lower})
        return (df)

    def get_prediction_interval(self,X=[]):
        """

        Chuck out the 95% prediction interval for the data passed

        Note that if X is a dataframe it may contain more columns than there are in the original data,
        therefore just pull out what we're after

        """
        #need to get the idempotent matrix
        i_n=numpy.matrix(numpy.ones(X.shape[0]))
        n_obs=X.shape[0]
        # M_0=numpy.matrix(numpy.eye(n_obs))-numpy.power(n_obs,-1)*i_n*i_n.T

        #Z is the X's without the offset
        # logging.debug(X.head())
        if isinstance(X,pd.core.frame.DataFrame):
            #assume its' called alpha
            X=X[self.independent_]



        df_pred=pd.DataFrame({'upper_pred':numpy.zeros(X.shape[0]),'lower_pred':numpy.zeros(X.shape[0])})
        df_pred['y_hat']=self.predict(X)
        df_pred['percent_ci']=0.0
        alpha=0.05
        t_val=stats.t.ppf(1-alpha/2,self.df_resid+1)
        for indx in df_pred.index:
            # print(df_pred.ix[indx].values[1:])
            # logging.debug(self.X_bar)
            # logging.debug(X.head())
            if "alpha" in self.independent_:
                x_0_x_bar=numpy.matrix(X.ix[indx].values[1:]-self.X_bar)
            else:
                x_0_x_bar=numpy.matrix(X.ix[indx].values-self.X_bar)
            
            
            # print(numpy.shape(x_0_x_bar))
            # print("************")
            se_e = self.s_y*numpy.sqrt(1 + (1/self.nobs) +
                x_0_x_bar*inv(self.Z_M_Z)*x_0_x_bar.T)

            df_pred.loc[indx,'upper_pred']=df_pred.loc[indx,'y_hat']+t_val*se_e
            df_pred.loc[indx,'lower_pred']=df_pred.loc[indx,'y_hat']-t_val*se_e

            df_pred.loc[indx,'percent_ci']=100*2*t_val*se_e/df_pred.loc[indx,'y_hat']
        return df_pred




def test_pred_interval(show_plot=False):
    from ml_ext import examples
    (coefs,df)=examples.gen_simplemodel_data(n=50,k=3)
    df.sort('X1',inplace=True)
    lr=LinModel()
    X=df[df.columns[df.columns!='y']]
    y=df.y


    lr.fit(X=X,y=y)
    lr.summary()
    df_ci=lr.get_confidence_interval_for_mean(X)
    df_pi=lr.get_prediction_interval(X)

    #Now use statsmodels to compare
    from statsmodels.sandbox.regression.predstd import wls_prediction_std
    import statsmodels.api as sm
    re = sm.OLS(y, X).fit()
    prstd, iv_l, iv_u = wls_prediction_std(re)

    if show_plot:
        (fig,ax)=plt.subplots(nrows=2,ncols=1,figsize=[14,12])

        cols=sns.color_palette('husl',n_colors=4)
        ax[0].scatter(X.X1,y,label='y',color=cols[3],alpha=0.4)
        
        ax[0].plot(X.X1,df_pi['upper_pred'],label='pred',color=cols[1],alpha=0.5)
        ax[0].plot(X.X1,df_pi['lower_pred'],color=cols[1],alpha=0.5)
        ax[0].plot(X.X1,df_ci['upper_mean'],color=cols[2],alpha=0.5)
        ax[0].plot(X.X1,df_ci['lower_mean'],label='mean_ci',color=cols[2],alpha=0.5)
        ax[0].scatter(X.X1,df_pi['y_hat'],label='y_hat',color=cols[0],alpha=0.5)
        ax[0].legend(loc='best')

        ax[1].scatter(X.X1,y,label='y',color=cols[3],alpha=0.4)
        ax[1].scatter(X.X1,df_ci['y_hat'],label='y_hat',color=cols[0],alpha=0.5)
        ax[1].plot(X.X1,iv_u,label='wls',color=cols[1],alpha=0.5)
        ax[1].plot(X.X1,iv_l,color=cols[1],alpha=0.5)
        ax[1].legend(loc='best')

    #get difference between uppers from each and check they are within 1%
    overall_diff=100*numpy.sum(iv_u-df_pi['upper_pred'])/numpy.sum(iv_u)
    logging.debug("Overall % difference in prediction ranges for upper bound: {}".format(overall_diff))
    assert overall_diff<0.1

def test_conf_interval_for_coefs():
    from ml_ext import examples
    (coefs,df)=examples.gen_simplemodel_data(n=50,k=1)
    #print(df.head())
    #df.sort('X0',inplace=True)
    lr=LinModel()
    X=df[df.columns[df.columns!='y']]
    y=df.y

    lr.fit(X=X,y=y)

    ci=lr.get_confidence_intervals_for_coefs()
    # print?(lr.se)
    import statsmodels.api as sm
    re = sm.OLS(y, X).fit()

    rmse=0
    for indx in ci.index.values:
        # print(re.conf_int().ix[indx,0])
        # print(ci.ix[indx,'lower'])
        rmse=rmse+numpy.power((re.conf_int().ix[indx,0]-ci.ix[indx,'lower'])/ci.ix[indx,'b'],2)

    assert 100*rmse/ci.shape[0]<0.1
    print("Error on confidence interval: {} %".format(100*rmse))

def test_data(df=[]):
    import statsmodels.api as sm
    print(df.describe())
    est = sm.OLS(df.y, df[df.columns[df.columns!='y']]).fit()
    print(est.summary())
    logging.info("Now for this algorithm")
    lr=LinModel()
    X=df[df.columns[df.columns!='y']]
    y=df.y

    lr.fit(X=X,y=y)
    lr.summary()