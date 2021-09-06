#!/usr/bin/env python
# coding: utf-8

# In[1]:


##Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error


# In[9]:


test=pd.read_csv("D:/DATA SCIENCE/Supervised Learning/Regression project practice/0000000000002419_test_ccpp_x_test.csv")


# In[10]:


test.shape


# In[13]:


test=test.drop(["Unnamed: 4"],axis=1)


# In[14]:


test.head(3)


# In[15]:


train=pd.read_excel("0000000000002419_training_ccpp_x_y_train.xlsx")


# In[16]:


train.shape


# In[17]:


train.iloc[7000:7176]


# In[18]:


train.isnull().sum()


# In[19]:


train.dtypes


# In[20]:


df=pd.concat([test,train],axis=0)
df.shape


# In[21]:


df.dtypes


# In[22]:


df.isnull().sum()


# In[23]:


df.Electrical_Hourly_Output.mean()


# In[24]:


df.Electrical_Hourly_Output.median()


# In[25]:


df["Electrical_Hourly_Output"]=df["Electrical_Hourly_Output"].fillna(df["Electrical_Hourly_Output"].mean())


# In[26]:


df.isnull().sum()


# In[27]:


for x in df:
    plt.hist(df[x],color="red")
    plt.xlabel(x)
    plt.ylabel("Frequency")
    plt.title("Distribution plot")
    plt.show()


# In[28]:


for x in df:
    sns.boxplot(df[x],color="yellow")
    plt.xlabel(x)
    plt.title("Outliers")
    plt.show()


# In[30]:


IQR_Amb=df.Ambient_Pressure.quantile(0.75)-df.Ambient_Pressure.quantile(0.25)
LL_Amb=df.Ambient_Pressure.quantile(0.25)-1.5*IQR_Amb
UL_Amb=df.Ambient_Pressure.quantile(0.75)+1.5*IQR_Amb

print("IQR of Ambient_Pressure: ",IQR_Amb)
print("LL of Ambient_Pressure: ",LL_Amb)
print("UL of Ambient_Pressure: ",UL_Amb)


# In[32]:


LL_Amb_Length=len(df[df.Ambient_Pressure<LL_Amb])
UL_Amb_Length=len(df[df.Ambient_Pressure>UL_Amb])
print(LL_Amb_Length)
print(UL_Amb_Length)


# In[33]:


UpperLimit_perc=UL_Amb_Length/len(df)
UpperLimit_perc


# In[34]:


LowerLimit_perc=LL_Amb_Length/len(df)
LowerLimit_perc


# In[35]:


df.shape


# In[105]:


columns=["TEMP","Exhasut_Vaccum","Ambient_Pressure","Electrical_Hourly_Output"]
sns.pairplot(df[columns],height=2,kind="scatter",diag_kind="kde")
plt.show()


# In[36]:


corrmat=df.corr()
corrmat


# In[110]:


sns.heatmap(corrmat,annot=True,vmax=1.0,vmin=-1.0,annot_kws={"size":12},cmap="bwr")
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.show()


# In[118]:


sns.scatterplot(x="Electrical_Hourly_Output",y="TEMP",data=df,hue="Electrical_Hourly_Output")
plt.show()


# In[116]:


X=df.drop(["Electrical_Hourly_Output"],axis=1)


# In[117]:


Y=df[["Electrical_Hourly_Output"]]


# In[118]:


x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=0)
from sklearn.preprocessing import StandardScaler
x_train=StandardScaler().fit_transform(x_train)
x_test=StandardScaler().fit_transform(x_test)
y_train=StandardScaler().fit_transform(y_train)
y_test=StandardScaler().fit_transform(y_test)


print("The shape of X-Train: ",x_train.shape)
print("The shape of X-Test: ",x_test.shape)
print("The shape of Y-Train: ",y_train.shape)
print("The shape of Y-Test: ",y_test.shape)


# In[119]:


from sklearn.linear_model import SGDRegressor
sgdr=SGDRegressor()
powerplant_SGD_model=sgdr.fit(x_train,y_train)


# In[120]:


SCORE1=sgdr.score(x_train,y_train)
print(SCORE1)


# In[121]:


powerplant_SGD_MODEL_predictions=powerplant_SGD_model.predict(x_test)
actual_powerplant_SGD_MODEL=y_test


# In[122]:


from statsmodels.tools.eval_measures import rmse
##rmse=rmse(actual_powerplant_SGD_MODEL,powerplant_SGD_MODEL_predictions)
MSE1=mean_squared_error(actual_powerplant_SGD_MODEL,powerplant_SGD_MODEL_predictions)
rmse=np.sqrt(MSE1)


# In[123]:


print(rmse)
print(MSE1)


# In[124]:


Columns=["MODEL","RMSE","SCORE"]
table=pd.DataFrame(columns=Columns)
Final=pd.Series({"MODEL":"PowerPlant SGD model",
                  "RMSE":rmse,
                   "SCORE":SCORE1,
                  })
table=table.append(Final,ignore_index=True)
table


# In[59]:


df.info()


# In[125]:


x=df.iloc[:,:4]


# In[126]:


x


# In[127]:


#convert it to numpy arrays
X=x.values


# In[69]:


from sklearn.preprocessing import scale


# In[128]:


#Scaling the values
X = scale(X)


# In[72]:


from sklearn.decomposition import PCA


# In[129]:


pca2 = PCA(n_components=4)
pca2.fit(X)


# In[130]:


#The amount of variance that each PC explains
var= pca2.explained_variance_ratio_
var


# In[131]:


#Cumulative Variance explains
var1=np.cumsum(np.round(pca2.explained_variance_ratio_, decimals=4)*100)
print(var1)
plt.plot(var1) # cumulative


# In[132]:


df1=df.copy()


# In[133]:


df1=df1.drop(["Ambient_Pressure","Relative _Humidity"],axis=1)


# In[134]:


df1.info()


# In[135]:


X=df1.drop(["Electrical_Hourly_Output"],axis=1)


# In[136]:


Y=df1[["Electrical_Hourly_Output"]]


# In[137]:


x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=0)
from sklearn.preprocessing import StandardScaler
x_train=StandardScaler().fit_transform(x_train)
x_test=StandardScaler().fit_transform(x_test)
y_train=StandardScaler().fit_transform(y_train)
y_test=StandardScaler().fit_transform(y_test)


print("The shape of X-Train: ",x_train.shape)
print("The shape of X-Test: ",x_test.shape)
print("The shape of Y-Train: ",y_train.shape)
print("The shape of Y-Test: ",y_test.shape)


# In[138]:


from sklearn.linear_model import SGDRegressor
sgdr=SGDRegressor()
powerplant_SGD_pcamodel=sgdr.fit(x_train,y_train)


# In[139]:


SCORE_PCA=sgdr.score(x_train,y_train)
print(SCORE_PCA)


# In[140]:


powerplant_SGD_PCA_MODEL_predictions=powerplant_SGD_pcamodel.predict(x_test)
actual_powerplant_SGD_PCA_MODEL=y_test


# In[141]:


from statsmodels.tools.eval_measures import rmse
##rmse=rmse(actual_powerplant_SGD_MODEL,powerplant_SGD_MODEL_predictions)
MSE_PCA=mean_squared_error(actual_powerplant_SGD_PCA_MODEL,powerplant_SGD_PCA_MODEL_predictions)
rmse_PCA=np.sqrt(MSE_PCA)


# In[142]:


print(rmse_PCA)
print(MSE_PCA)


# In[143]:


Final=pd.Series({"MODEL":"PowerPlant SGD PCA model",
                  "RMSE":rmse_PCA,
                   "SCORE":SCORE_PCA,
                  })
table=table.append(Final,ignore_index=True)
table


# In[144]:


from statsmodels.stats.outliers_influence import variance_inflation_factor
Vif=pd.DataFrame()
Vif["VIF FACTOR"]= [variance_inflation_factor(df.values,i)for i in range(df.shape[1])]
Vif["FEATURES"]=df.columns
Vif.sort_values("VIF FACTOR",ascending=False).reset_index(drop=True)


# In[145]:


powerplant_DF=df.drop(["Ambient_Pressure","Exhasut_Vaccum",],axis=1)


# In[146]:


powerplant_DF.info()


# In[147]:


X=powerplant_DF.drop(["Electrical_Hourly_Output"],axis=1)


# In[148]:


Y=powerplant_DF[["Electrical_Hourly_Output"]]


# In[149]:


x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=0)
from sklearn.preprocessing import StandardScaler
x_train=StandardScaler().fit_transform(x_train)
x_test=StandardScaler().fit_transform(x_test)

print("The shape of X-Train: ",x_train.shape)
print("The shape of X-Test: ",x_test.shape)
print("The shape of Y-Train: ",y_train.shape)
print("The shape of Y-Test: ",y_test.shape)


# In[150]:


from sklearn.linear_model import SGDRegressor
sgdr=SGDRegressor()
powerplant_SGD2_model=sgdr.fit(x_train,y_train)


# In[151]:


SCORE_VIF=sgdr.score(x_train,y_train)
SCORE_VIF


# In[172]:


powerplant_SGD2_model_predictions=powerplant_SGD2_model.predict(x_test)
actual_powerplant_SGD2_model=(y_test)


# In[153]:


MSE3=mean_squared_error(actual_powerplant_SGD2_model,powerplant_SGD2_model_predictions)
rmse3=np.sqrt(MSE3)


# In[154]:


rmse3


# In[155]:


Final=pd.Series({"MODEL":"PowerPlant SGD3 model",
                  "RMSE":rmse3,
                   "SCORE":SCORE1,
                  })
table=table.append(Final,ignore_index=True)
table


# In[173]:


Predicted_Power=powerplant_SGD2_model_predictions
print(Predicted_Power)

