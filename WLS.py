import pandas
import matplotlib.pyplot as plt
import numpy
import seaborn
from statsmodels.sandbox.regression.predstd import wls_prediction_std
import statsmodels.api as stm

location = 'index.csv'
df = pandas.read_csv(location)

# Y = df.Cost
# X = df.Responses


#Check the Homoscedasticity
# In regression analysis heteroscedasticity means a situation in which the variance of the dependent variable (Y) varies across the levels of the independent data (X). Heteroscedasticity can complicate analysis because regression analysis is based on an assumption of equal variance across the levels of the independent data. 
fig, ax = plt.subplots(figsize=(8, 4))
ax.scatter(df.Responses, df.Cost, alpha=0.5, color='orchid')
fig.suptitle('Example Scatter Plot')
fig.tight_layout(pad=2); 
ax.grid(True)
fig.savefig('FFFFFFFFFFFFF.png', dpi=125)

#first OLS model
ols_mod = stm.OLS(df.Cost,df.Responses).fit()
summary = ols_mod.summary()

fig, ax = plt.subplots(figsize=(8, 4))
ax.scatter(df.Responses,ols_mod.resid, alpha=0.5, color='orchid')
fig.suptitle('ResponsesVSresid Scatter Plot')
fig.tight_layout(pad=2); 
ax.grid(True)
fig.savefig('ResponsesVSresid.png', dpi=125)


# build a regression model of the standard deviation against Cost. We do that by regressing the absolute values of the residuals against Cost, since the absolute residuals are an estimator of the standard deviation of Responses at different values of Cost.
get_residual = ols_mod.resid
# absolute_values_of_the_residuals = ols_mod.resid.abs()

#ScatterPlot ResponsesVSAbsres
fig, ax = plt.subplots(figsize=(8, 4))
ax.scatter(df.Responses,get_residual, alpha=0.5, color='orchid')
fig.suptitle('ResponsesVSAbsresid Scatter Plot')
fig.tight_layout(pad=2); 
ax.grid(True)
fig.savefig('ResponsesVSAbsresid.png', dpi=125) 

ols_mod_with_residuals = stm.OLS(get_residual,df.Responses).fit()
residuals_predict_values = ols_mod_with_residuals.predict()
# numpy.array([1/each*each for each in residuals_predict_values]
wls_mod = stm.WLS(df.Cost,df.Responses,weights=[1/each*each for each in get_residual]).fit()
fig, ax = plt.subplots(figsize=(8, 4))
x,y = ols_mod.predict(),wls_mod.predict()
ax.scatter(df.Responses, df.Cost, alpha=0.5, color='orchid')
ax.scatter(x,y)
fig.suptitle('Example Scatter Plot')
fig.tight_layout(pad=2); 
ax.grid(True)
fig.savefig('olsVSwls_fitted.png', dpi=125)
