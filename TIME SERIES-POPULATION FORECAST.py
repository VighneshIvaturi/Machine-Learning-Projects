#!/usr/bin/env python
# coding: utf-8

# ### IMPORTING REQUIRED LIBRARIES

# In[62]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.tsa.stattools import adfuller
get_ipython().run_line_magic('matplotlib', 'inline')
from math import sqrt
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.seasonal import seasonal_decompose
import seaborn as sns
from statsmodels.tsa.stattools import adfuller


# #### READING THE DATA

# In[38]:


df=pd.read_csv("D:/POP.csv")


# In[4]:


df.head()


# In[10]:


df.info()
df.shape


# In[39]:


df.isnull().sum()


# In[7]:


df["date"].head()


# In[8]:


df["date"].describe()


# In[41]:


df['Date'] = pd.to_datetime(df['date'])
df['year'] = pd.DatetimeIndex(df['date']).year
df['month'] = pd.DatetimeIndex(df['date']).month
df.info()


# In[42]:


#Indexing data with Order_Date
df = df.set_index('Date')
df.head()


# In[16]:


df["value"].describe()


# In[21]:


df["value"].plot(kind='hist',facecolor="blue")


# In[22]:


df.groupby("year")["value"].describe()
        


# In[25]:


df.info()


# In[43]:


df=df.drop(["realtime_start","realtime_end",'year',"month"],axis=1)


# In[45]:


df=df.drop(["date"],axis=1)


# In[46]:


df.head()


# In[28]:


df.describe()


# #### LINE PLOT OF ORIGINAL DATA

# In[29]:


plt.plot(df,"r")


# In[146]:


df.plot(kind='hist',facecolor="green")
plt.title("Histogram of orginal data")


# In[31]:


df.plot(kind='kde',color="black")


# In[35]:


props2 = dict(boxes = 'red', whiskers = 'green', medians = 'black', caps = 'red')
df["value"].plot.box(color=props2, patch_artist = True, vert = False) #Outliers


# #### DECOMPOSING THE TIME SERIES

# In[47]:


df_decomp_mul=seasonal_decompose(df,period=1,model="mul")
df_decomp_mul.plot()


# In[48]:


df_decomp_mul=seasonal_decompose(df,period=1,model="additive")
df_decomp_mul.plot()


# #### From the above decomposition we can observe that there is no trend and seasonality in the data

# #### MOVING AVERAGE METHOD

# In[100]:


dfMA = df.rolling(window=12).mean()


# In[136]:


plt.plot(df,label='Original data')
plt.plot(dfMA,color='red',label='Rolling avg')
plt.legend(loc='best')
plt.show()


# ##### RESIDUALS

# In[101]:


df_ma_res = df - dfMA
df_ma_res.head()


# In[137]:


df_ma_res=df_ma_res.dropna()
df_ma_res.head()


# In[138]:


#Lineplot
df_ma_res.plot()
plt.title('Line plot of Residuals')


# In[148]:


df_ma_res.plot(kind='hist',facecolor="blue")
plt.title("Histogram of residuals")


# In[140]:


#squaring residuals/errors
df_ma_res_se=pow(df_ma_res,2)
print(df_ma_res_se.head())

#mean of squared errors
df_ma_res_mse=df_ma_res_se.sum()/len(df_ma_res_se)
print("mse: ",df_ma_res_mse)

df_ma_res_rmse=sqrt(df_ma_res_mse)
print("rmse: ",df_ma_res_rmse)


# In[59]:


rolmean = df.rolling(window=12).mean()
rolstd = df.rolling(window=12).std()

plt.plot(rolmean,color='red',label='Rolling avg')
plt.plot(rolstd,color='black',label='Rolling std')
plt.legend(loc='best')
plt.show()


# #### STATIONARITY TEST

# In[107]:


df_adf=adfuller(df,autolag='AIC')


# In[60]:


print('ADF Statistic: %f' % df_adf[0])
print('p-value: %f' % df_adf[1])
print('Critical Values:')
for key, value in df_adf[4].items():
    print('\t%s: %.3f' % (key, value))


# #### p-value: 0.85 ie > 0.05, Null Hypothesis is accepted, so, Data is not stationary
# H0 data is not stationary

# ### AUTO ARIMA

# In[141]:


from pmdarima import auto_arima


# In[142]:


df_mod=auto_arima(df)
df_mod.summary()


# In[143]:


df_RES = pd.DataFrame(df_mod.resid(), index=df.index)
df_RES 


# In[144]:


#Histogram of residuals
plt.hist(df_res)
plt.title('Histogram of Residuals')


# In[84]:


#Plotting acf & pacf for residuals
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
plot_acf(df_RES, lags=20) 
plot_pacf(df_RES, lags=20)


# In[145]:


#squaring residuals/errors
df_RES_se=pow(df_RES,2)
print(df_RES_se.head())

#mean of squared errors
df_RES_mse=df_RES_se.sum()/len(df_RES_se)
print("mse: ",df_RES_mse)

df_RES_rmse=sqrt(df_RES_mse)
print("rmse: ",df_RES_rmse)


# In[133]:


#Plot comparision Actual, Model Values & Residuals
plt.plot(df)
plt.plot(df_RES, 'r')
plt.legend(['Actual', 'Residuals'],
           bbox_to_anchor=(1, 1), loc=2)
plt.xticks(rotation=45)
plt.show()


# In[117]:


#Predict
df_pred = df_mod.predict(n_periods=121) 
df_pred = pd.DataFrame(df_pred, 
                             index=pd.date_range(start='2020-01-01',end='2030-01-01', freq='MS'))


# In[128]:


#Plot comparision Actual, Model Values & Forecast
plt.plot(df)
plt.plot(df_mod_v)
plt.plot(df_pred)
plt.legend(['Actual', "Fitted",'Forecast'],
           bbox_to_anchor=(1, 1), loc=2)
plt.xticks(rotation=45)
plt.show()


# ## This model clearly fits very well in the timeseries and gets the prediction for the next 10 years from 2020-01-01.

# ##### POPULATION FORECAST FOR 10 YEARS 

# In[135]:


df_pred

