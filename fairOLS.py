import pandas
import matplotlib.pyplot as plt
import statsmodels.api as stm
import numpy
import seaborn
from statsmodels.sandbox.regression.predstd import wls_prediction_std

location = 'datasets/fairOLS.csv'
df = pandas.read_csv(location)
endog = df.affairs
exog = df[['rate_marriage','age','yrs_married','children','religious','educ','occupation_husb']]

# --------------------- CHECK HETROSCEDASTICITY -------------------
DataFrame_to__array = numpy.array(exog)
DataFrame_to__array = DataFrame_to__array.flatten()[:len(endog)]
fig, ax = plt.subplots(figsize=(8, 4))
ax.scatter(endog, DataFrame_to__array.flatten()[:len(endog)] , alpha=0.5, color='orchid')
fig.suptitle('Example Scatter Plot')
fig.tight_layout(pad=2); 
ax.grid(True)
fig.savefig('CCCCCCCCCCCCCCC.png', dpi=125)
mod = stm.OLS(endog,exog).fit()
# mod.summary()
# fitted_values = mod.fittedvalues
# get_parameters = mod.params

# plt.figure(figsize=(10,10))
# plt.legend(loc='best');
# plt.plot(df.affairs[:20],df[['rate_marriage','age','yrs_married','children','religious','educ','occupation_husb']][:20] ,'o',label='data')
# plt.plot(mod.fittedvalues ,df.affairs,'o',color='black',label='fitted')
# plt.savefig('fairOLS',format='png')

# fig, ax = plt.subplots(figsize=(8,6))
# ax.legend(loc='best');
# ax.plot(numpy.linspace(df.affairs.min(),df.affairs.max(),30),numpy.linspace(df.rate_marriage.min(),df.rate_marriage.max(),30),'o', alpha=0.5,label="data")
# ax.plot(numpy.linspace(df.affairs.min(),df.affairs.max(),30),numpy.linspace(mod.fittedvalues.min(),mod.fittedvalues.max(),30),'o', alpha=0.5,label="fitted")
# fig.savefig('KKKKKKKKKKK')

prstd, iv_l, iv_u = wls_prediction_std(mod)
fig, ax = plt.subplots(figsize=(8,6))
ax.plot(df.affairs[:20] , df[['rate_marriage','age','yrs_married','children','religious','educ','occupation_husb']][:20] , 'o', label="data")
ax.plot(df.affairs[:20], mod.fittedvalues[:20] , 'r--.', label="OLS")
ax.plot(df.affairs[:20], iv_u, 'r--')
ax.plot(df.affairs[:20], iv_l, 'r--')
ax.legend(loc='best');

# fig, ax = plt.subplots(figsize=(8, 4))
# ax.scatter(data.endog, data.exog.flatten()[:len(data.endog)] , alpha=0.5, color='orchid')
# fig.suptitle('Example Scatter Plot')
# fig.tight_layout(pad=2); 
# ax.grid(True)
# fig.savefig('CCCCCCCCCCCCCCC.png', dpi=125)