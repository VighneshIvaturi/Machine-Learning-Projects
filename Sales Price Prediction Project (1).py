#!/usr/bin/env python
# coding: utf-8

# # Project Title-Stores Sales Prediction

# ### Domain Sales & Marketing
# 

# ## Problem Statement

# #### Nowadays, shopping malls and Big Marts keep track of individual item sales data in order to forecast future client demand and adjust inventory management. In a data warehouse, these data stores hold a significant amount of consumer information and particular item details. By mining the data store from the data warehouse, more anomalies and common patterns can be discovered.

# # Data Definition

# Item_Identifier : Unique product ID
# 
# Item_Weight: Weight of product
# 
# Item_Fat_Content: Whether the product is low fat or not
# 
# Item_Visibility: The % of total display area of all products in a store allocated to the particular product
# 
# Item_Type: The category to which the product belongs
# 
# Item_MRP: Maximum Retail Price (list price) of the product
# 
# Outlet_Identifier: Unique store ID
# 
# Outlet_Establishment_Year: The year in which store was established
# 
# Outlet_Size: The size of the store in terms of ground area covered
# 
# Outlet_Location_Type: The type of city in which the store is located
# 
# Outlet_Type: Whether the outlet is just a grocery store or some sort of supermarket
# 
# Item_Outlet_Sales: Sales of the product in the particulat store. This is the outcome variable to be predicted.

# ### 1. Import Libraries

# In[1]:


##Libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error


# In[2]:


from warnings import filterwarnings
filterwarnings("ignore")


# ### 2. Read Data

# In[35]:


Sales_Price_df=pd.read_csv("D:/DATA SCIENCE/I NEURON/STORE SALES PREDICTION/Train.csv")


# In[4]:


Sales_Price_df.head()


# #### Dimensions of the data

# In[5]:


Sales_Price_df.info()
Sales_Price_df.shape


# ### 3.Data Analysis and Preparation

# #### Null Values

# In[6]:


Sales_Price_df.isnull().sum()


# #### Statistical Description of data

# In[7]:


Sales_Price_df.describe()


# #### Filling Null Values

# In[8]:


Sales_Price_df["Item_Weight"].mean()


# In[9]:


Sales_Price_df["Item_Weight"].median()


# In[36]:


Sales_Price_df["Item_Weight"]=Sales_Price_df["Item_Weight"].fillna(Sales_Price_df["Item_Weight"].mean())


# In[11]:


Sales_Price_df["Outlet_Size"].value_counts()


# In[12]:


Sales_Price_df["Outlet_Size"].mode()


# In[37]:


Sales_Price_df["Outlet_Size"]=Sales_Price_df["Outlet_Size"].fillna("Medium")


# In[38]:


Sales_Price_df.isnull().sum()


# ### Analysing Target Variable

# In[15]:


##Histogram
plt.hist(Sales_Price_df.Item_Outlet_Sales,color='red')
plt.xlabel('Item outlet sales')
plt.ylabel('Frequency')
plt.title('Histogram-Item outlet sales')
plt.show


# In[16]:


Sales_Price_df["Item_Outlet_Sales"].describe()


# ### From the above statistical description we can clearly understnad There is a huge difference of Sales at minimum,1st qunatile, 3rd Quantile and maximum.

# #### Box Plot

# In[17]:


props2 = dict(boxes = 'red', whiskers = 'green', medians = 'black', caps = 'red')
Sales_Price_df.Item_Outlet_Sales.plot.box(color=props2, patch_artist = True, vert = False)


# #### Outliers Identification

# In[18]:


IQR=Sales_Price_df.Item_Outlet_Sales.quantile(0.75)-Sales_Price_df.Item_Outlet_Sales.quantile(0.25)
LL=Sales_Price_df.Item_Outlet_Sales.quantile(0.25)-(1.5*IQR)
UL=Sales_Price_df.Item_Outlet_Sales.quantile(0.75)+(1.5*IQR)

print("IQR: ",IQR)
print("LL: ",LL)
print("UL: ",UL)


# In[20]:


up_lim_len=len(Sales_Price_df.Item_Outlet_Sales[Sales_Price_df.Item_Outlet_Sales>UL])


# In[22]:


UpperLimit_perc=up_lim_len/len(Sales_Price_df)
UpperLimit_perc


# In[23]:


Sales_Price_df.Item_Outlet_Sales.value_counts()
print(len(Sales_Price_df.Item_Outlet_Sales[Sales_Price_df.Item_Outlet_Sales <0])) 
print(len(Sales_Price_df.Item_Outlet_Sales[Sales_Price_df.Item_Outlet_Sales >1000]))
print(len(Sales_Price_df.Item_Outlet_Sales[Sales_Price_df.Item_Outlet_Sales >2000])) 
print(len(Sales_Price_df.Item_Outlet_Sales[Sales_Price_df.Item_Outlet_Sales >7000])) 


# In[120]:


sns.heatmap(cormat,annot=True,vmax=+1,vmin=-1,annot_kws={"size":11.5})
plt.title("Correlation Matrix")
plt.show()


# #### Value counts 

# In[73]:


for i in range(0,12):
    print(Sales_Price_df.iloc[:,i].value_counts())
    print("*"*10)


# In[24]:


from scipy.stats import f_oneway


# In[25]:


import statsmodels.api as sm


# In[26]:


from statsmodels.formula.api import ols


# In[34]:


Sales_Price_df.info()
Sales_Price_df.shape


# ### 1.Item_Identifier

# In[122]:


Sales_Price_df.Item_Identifier.describe()


# ### FDW13 is the most repeated Item Identifier number. With frequency of 10.

# In[261]:


Sales_Price_df.Item_Identifier.value_counts()


# In[262]:


Sales_Price_df.groupby("Item_Identifier")["Item_Outlet_Sales"].describe()


# ### 2.Item_Weight

# In[263]:


Sales_Price_df.Item_Weight.value_counts()


# In[264]:


Sales_Price_df.groupby("Item_Weight")["Item_Outlet_Sales"].describe()


# #### SCATTER PLOT

# In[45]:


sns.scatterplot(x="Item_Weight",y="Item_Outlet_Sales",hue="Item_Outlet_Sales",data=Sales_Price_df)
plt.title("Weight Vs Outlet Sales")
plt.show()


# In[60]:


Sales_Price_df.Item_Weight.describe()


# #### Histogram

# In[79]:


##Histogram
plt.hist(Sales_Price_df.Item_Weight,color='blue')
plt.title('Histogram-Item Weight')
plt.show


# #### Box Plot

# In[47]:


props2 = dict(boxes = 'red', whiskers = 'green', medians = 'black', caps = 'red')
Sales_Price_df.Item_Weight.plot.box(color=props2, patch_artist = True, vert = False)


# #### Outliers Identification

# In[46]:


IQR=Sales_Price_df.Item_Weight.quantile(0.75)-Sales_Price_df.Item_Weight.quantile(0.25)
LL=Sales_Price_df.Item_Weight.quantile(0.25)-(1.5*IQR)
UL=Sales_Price_df.Item_Weight.quantile(0.75)+(1.5*IQR)

print("IQR: ",IQR)
print("LL: ",LL)
print("UL: ",UL)


# In[61]:


len(Sales_Price_df.Item_Weight[Sales_Price_df.Item_Weight>UL])


# ### 3.Item_Fat_Content

# In[62]:


Sales_Price_df.Item_Fat_Content.describe()


# #### The most repeated selling product is the item with Low Fat content and has frequency of 5012 out of 8398 observations.

# In[63]:


Sales_Price_df.groupby("Item_Fat_Content")["Item_Outlet_Sales"].describe()


# ### Update types of Item_Fat_Content
# 
# We found typing error and difference in representation in categories of Item_Fat_Content variable.

# In[39]:


Sales_Price_df['Item_Fat_Content'] =Sales_Price_df['Item_Fat_Content'].replace({'LF':'Low Fat',
                                                             'reg':'Regular',
                                                             'low fat':'Low Fat'})
print(Sales_Price_df['Item_Fat_Content'].value_counts())


# #### COUNTPLOT

# In[22]:


sns.countplot(x='Item_Fat_Content',data=Sales_Price_df)
plt.xlabel=("Item_Fat_Content")
plt.ylabel=("Counts")
#plt.rcParams['figure.figsize'] = [10.8]
plt.title("Item_Fat_Content counting")


# In[40]:


Sales_Price_df.info()


# In[44]:


plt.plot(Sales_Price_df[Sales_Price_df.Outlet_Establishment_Year==1985].groupby('Item_Fat_Content')['Item_Outlet_Sales'].sum())
plt.plot(Sales_Price_df[Sales_Price_df.Outlet_Establishment_Year==1987].groupby('Item_Fat_Content')['Item_Outlet_Sales'].sum())
plt.plot(Sales_Price_df[Sales_Price_df.Outlet_Establishment_Year==1999].groupby('Item_Fat_Content')['Item_Outlet_Sales'].sum())
plt.plot(Sales_Price_df[Sales_Price_df.Outlet_Establishment_Year==1997].groupby('Item_Fat_Content')['Item_Outlet_Sales'].sum())
plt.plot(Sales_Price_df[Sales_Price_df.Outlet_Establishment_Year==2004].groupby('Item_Fat_Content')['Item_Outlet_Sales'].sum())
plt.plot(Sales_Price_df[Sales_Price_df.Outlet_Establishment_Year==2002].groupby('Item_Fat_Content')['Item_Outlet_Sales'].sum())
plt.plot(Sales_Price_df[Sales_Price_df.Outlet_Establishment_Year==2009].groupby('Item_Fat_Content')['Item_Outlet_Sales'].sum())
plt.plot(Sales_Price_df[Sales_Price_df.Outlet_Establishment_Year==2007].groupby('Item_Fat_Content')['Item_Outlet_Sales'].sum())
plt.plot(Sales_Price_df[Sales_Price_df.Outlet_Establishment_Year==1998].groupby('Item_Fat_Content')['Item_Outlet_Sales'].sum())
plt.xlabel(['Item_Fat_Content'])
plt.ylabel('Sales')
plt.title('Sum of Sales Item_Fat_Content wise in respective years')
plt.legend([1985,1987,1999,1997,2004,2002,2009,2007,1998])
plt.show()


# #### BARPLOT

# In[23]:


Sales_Price_df.groupby("Item_Fat_Content")["Item_Outlet_Sales"].mean().plot(kind="bar")
plt.xlabel="Item_Fat_Content"
plt.ylabel="Item_Outlet_Sales"
plt.show


# #### PIE CHART

# In[24]:


Frequency_by_Item=Sales_Price_df["Item_Fat_Content"].value_counts()
keys=Frequency_by_Item.keys().to_list()
counts=Frequency_by_Item.to_list()

plt.pie(x=counts,labels=keys,autopct="%1.1f%%")
circle=plt.Circle(xy=(0,0),radius=0.2,color="white")
plt.gcf()
plt.gca().add_artist(circle)
plt.title("Item_Fat_Content Distribution")
plt.show()
        


# ### 4.Item_Visibility

# In[266]:


Sales_Price_df.Item_Visibility.value_counts()


# In[267]:


Sales_Price_df.Item_Visibility.describe()


# ### Item_Visibility has a min value of zero. This makes no practical sense because when a product is being sold in a store, the visibility cannot be 0.

# ### Update Item_Visibility
# 
# We noticed that the minimum value in visibility is 0, which makes not practical. Lets consider it like missing information and replace it with mean visibility of that product.

# In[45]:


print(Sales_Price_df['Item_Visibility'].mean())    
   
Sales_Price_df['Item_Visibility'] = Sales_Price_df['Item_Visibility'].replace(0, 0.06978024245414721)
print ('Number of 0 values after modification: %d'%sum(Sales_Price_df['Item_Visibility'] == 0))


# In[11]:


Sales_Price_df.Item_Visibility.describe()


# #### HISTOGRAM

# In[126]:


##Histogram
plt.hist(Sales_Price_df.Item_Visibility,color='blue')
plt.title('Histogram-Item Visibility')
plt.show


# #### BOX PLOT

# In[55]:


props2 = dict(boxes = 'red', whiskers = 'green', medians = 'black', caps = 'red')
Sales_Price_df.Item_Visibility.plot.box(color=props2, patch_artist = True, vert = False)


# #### SCATTER PLOT

# In[56]:


sns.scatterplot(x="Item_Visibility",y="Item_Outlet_Sales",hue="Item_Outlet_Sales",data=Sales_Price_df)
plt.title("Visibility Vs Outlet Sales")
plt.show()


# ### 5.Item_Type

# In[63]:


Sales_Price_df.groupby("Item_Type")["Item_Outlet_Sales"].describe()


# ### Grouping Item type
# We can observe that the Item_Type variable has 16 categories. If you look at the Item_Identifier, i.e. the unique ID of each item, it starts with either FD, DR or NC. If you see the categories, these look like being Food, Drinks and Non-Consumables. So I added the Item_Type_Identifier variable column:

# In[47]:


Sales_Price_df['Item_Type_Combined'] = Sales_Price_df['Item_Identifier'].apply(lambda x: x[0:2])

Sales_Price_df['Item_Type_Combined'] = Sales_Price_df['Item_Type_Combined'].map({'FD':'Food',
                                                             'NC':'Non-Consumable',
                                                             'DR':'Drinks'})
Sales_Price_df['Item_Type_Combined'].value_counts()


# In[48]:


plt.plot(Sales_Price_df[Sales_Price_df.Outlet_Establishment_Year==1985].groupby('Item_Type_Combined')['Item_Outlet_Sales'].sum())
plt.plot(Sales_Price_df[Sales_Price_df.Outlet_Establishment_Year==1987].groupby('Item_Type_Combined')['Item_Outlet_Sales'].sum())
plt.plot(Sales_Price_df[Sales_Price_df.Outlet_Establishment_Year==1999].groupby('Item_Type_Combined')['Item_Outlet_Sales'].sum())
plt.plot(Sales_Price_df[Sales_Price_df.Outlet_Establishment_Year==1997].groupby('Item_Type_Combined')['Item_Outlet_Sales'].sum())
plt.plot(Sales_Price_df[Sales_Price_df.Outlet_Establishment_Year==2004].groupby('Item_Type_Combined')['Item_Outlet_Sales'].sum())
plt.plot(Sales_Price_df[Sales_Price_df.Outlet_Establishment_Year==2002].groupby('Item_Type_Combined')['Item_Outlet_Sales'].sum())
plt.plot(Sales_Price_df[Sales_Price_df.Outlet_Establishment_Year==2009].groupby('Item_Type_Combined')['Item_Outlet_Sales'].sum())
plt.plot(Sales_Price_df[Sales_Price_df.Outlet_Establishment_Year==2007].groupby('Item_Type_Combined')['Item_Outlet_Sales'].sum())
plt.plot(Sales_Price_df[Sales_Price_df.Outlet_Establishment_Year==1998].groupby('Item_Type_Combined')['Item_Outlet_Sales'].sum())
plt.xlabel(['Item_Type_Combined'])
plt.ylabel('Sales')
plt.title('Sum of Sales Item Type wise in respective years')
plt.legend([1985,1987,1999,1997,2004,2002,2009,2007,1998])
plt.show()


# #### COUNT PLOT

# In[31]:


sns.countplot(x="Item_Type_Combined", data=Sales_Price_df)
plt.rcParams['figure.figsize'] = [10,5]
plt.xlabel=("Item Type")
plt.ylabel=('counts')
plt.title("Item type wise counting")


# #### PIE CHART

# In[36]:


Frequency_by_Item=Sales_Price_df["Item_Type_Combined"].value_counts()


# In[37]:


keys=Frequency_by_Item.keys().to_list()
counts=Frequency_by_Item.to_list()


# In[38]:



plt.pie(x=counts,labels=keys,autopct="%1.1f%%")
circle=plt.Circle(xy=(0,0),radius=0.4,color="white")
plt.gcf()
plt.gca().add_artist(circle)
plt.title("Item Type Distribution")
plt.show()
        


# ##### Food Items are most sold item types.

# ### 6.Outlet_Identifier

# In[84]:


Sales_Price_df.Outlet_Identifier.describe()


# In[86]:


Sales_Price_df.groupby("Outlet_Identifier")["Item_Outlet_Sales"].describe()


# #### OUT019 and OUT010 are having less outlet sales among 10 Outlet identifiers.

# #### BAR PLOT

# In[94]:


Sales_Price_df.groupby("Outlet_Identifier")["Item_Outlet_Sales"].mean().plot(kind="bar")
plt.xlabel="Outlet_Identifier"
plt.ylabel="Item_Outlet_Sales"
plt.show


# #### COUNT PLOT

# In[109]:


sns.countplot(x='Outlet_Identifier',data=Sales_Price_df)
plt.xlabel=("Item_Fat_Content")
plt.ylabel=("Counts")
plt.rcParams['figure.figsize'] = [8,10]
plt.title("Outlet_Identifier counting")


# In[143]:


plt.plot(Sales_Price_df.groupby("Outlet_Identifier")["Item_Outlet_Sales"].mean())
plt.rcParams['figure.figsize'] = [5,5]
plt.xticks(rotation=90)

plt.show()


# ### 7.Item_MRP

# In[108]:


Sales_Price_df.Item_MRP.value_counts()


# #### Histogram

# In[120]:


##Histogram
plt.hist(Sales_Price_df.Item_MRP,color='blue')
plt.xlabel=("Item_MRP")
plt.title('Histogram-Item_MRP')
plt.rcParams['figure.figsize'] = [10,5]
plt.show


# #### BOX PLOT

# In[328]:


props2 = dict(boxes = 'red', whiskers = 'green', medians = 'black', caps = 'red')
Sales_Price_df.Item_MRP.plot.box(color=props2, patch_artist = True, vert = False)


# #### SCATTER PLOT

# In[121]:


sns.scatterplot(x="Item_MRP",y="Item_Outlet_Sales",hue="Item_Outlet_Sales",data=Sales_Price_df)
plt.title("MRP Vs Outlet Sales")
plt.rcParams['figure.figsize'] = [10,5]
plt.show()


# ### 8.Outlet_Establishment_Year

# In[14]:


Sales_Price_df.Outlet_Establishment_Year.value_counts()


# In[18]:


Sales_Price_df.groupby("Outlet_Establishment_Year")["Item_Outlet_Sales"].describe()


# #### Outlet_Establishment_Years vary from 1985 to 2009. The values might not be apt in this form. Rather, if we can convert them to how old the particular store is, it should have a better impact on sales.

# In[122]:


Sales_Price_df.Outlet_Establishment_Year.describe()


# #### BOX PLOT

# In[124]:


props2 = dict(boxes = 'red', whiskers = 'green', medians = 'black', caps = 'red')
Sales_Price_df.Outlet_Establishment_Year.plot.box(color=props2, patch_artist = True, vert = False)


# #### COUNT PLOT

# In[125]:


sns.countplot(x='Outlet_Establishment_Year',data=Sales_Price_df)
plt.xlabel=("Outlet Establishment Year")
plt.ylabel=('Counts')
plt.rcParams['figure.figsize'] = [5,10]
plt.title("Outlet_Establishment_Year counting")


# In[150]:


plt.plot(Sales_Price_df.groupby("Outlet_Establishment_Year")["Item_Outlet_Sales"].mean())
plt.rcParams['figure.figsize'] = [5,2]
plt.xticks(rotation=90)

plt.show()


# ### We shall add column having values of stores working years as "Outlet_Years"

# In[49]:


Sales_Price_df['Outlet_Years'] = 2009 - Sales_Price_df['Outlet_Establishment_Year']
Sales_Price_df['Outlet_Years'].describe()


# In[50]:


sns.countplot(x='Outlet_Years',data=Sales_Price_df)
plt.xlabel=("Outlet_Years")
plt.ylabel=('Counts')
plt.rcParams['figure.figsize'] = [5,10]
plt.title("Outlet_Years counting")


# In[41]:


plt.figure(figsize = (12,6))
ax = sns.boxplot(x = 'Outlet_Years', y = 'Item_Outlet_Sales', data = Sales_Price_df)
ax.set_xticklabels(ax.get_xticklabels(), rotation = 45)
ax.set_title('Outlet years vs Item_Outlet_Sales')
ax.set_xlabel('', fontsize = 15)
ax.set_ylabel('Item_Outlet_Sales', fontsize = 15)

plt.show()


# In[127]:


Sales_Price_df.Outlet_Size.value_counts()
Sales_Price_df.Outlet_Size.describe()


# ### 9.Outlet_Size

# In[125]:


Sales_Price_df.groupby("Outlet_Size")["Item_Outlet_Sales"].describe()


# #### BAR PLOT

# In[136]:


Sales_Price_df.groupby("Outlet_Size")["Item_Outlet_Sales"].mean().plot(kind="bar")
plt.xlabel="Outlet_Size"
plt.ylabel="Item_Outlet_Sales"
plt.show
plt.rcParams['figure.figsize'] = [5,5]


# Let us assign numerical values to Outlet_Size

# In[16]:


#Creating dict file
Outlet_Size = {'High':1, 'Medium':2, 'Small':3}
#Converting Size names to numbers
Sales_Price_df.Outlet_Size = [Outlet_Size[item] for item in Sales_Price_df.Outlet_Size]


# In[284]:


Sales_Price_df.info()


# In[285]:


Sales_Price_df.groupby("Outlet_Size")["Item_Outlet_Sales"].describe()


# ### 10.Outlet_Location_Type

# In[286]:


Sales_Price_df.Outlet_Location_Type.describe()


# In[152]:


Sales_Price_df.groupby("Outlet_Location_Type")["Item_Outlet_Sales"].describe()


# #### BAR PLOT

# In[153]:


Sales_Price_df.groupby("Outlet_Location_Type")["Item_Outlet_Sales"].mean().plot(kind="bar")
plt.xlabel="Outlet_Location_Type"
plt.ylabel="Item_Outlet_Sales"
plt.show
plt.rcParams['figure.figsize'] = [5,5]


# Let us assign numerical values to Outlet_Location_Type

# In[17]:


#Creating dict file
Outlet_Location_Type = {'Tier 1':1, 'Tier 2':2, 'Tier 3':3}
#Converting Size names to numbers
Sales_Price_df.Outlet_Location_Type = [Outlet_Location_Type[item] for item in Sales_Price_df.Outlet_Location_Type]


# ### 11.Outlet_Type

# In[133]:


Sales_Price_df.Outlet_Type.describe()


# In[155]:


Sales_Price_df.groupby("Outlet_Type")["Item_Outlet_Sales"].describe()


# #### COUNT PLOT

# In[164]:


sns.countplot(x='Outlet_Type',data=Sales_Price_df)
plt.xlabel=("Outlet_Type")
plt.ylabel=('Counts')
plt.rcParams['figure.figsize'] = [10,5]
plt.title("Outlet_Type counting")


# #### BAR PLOT

# In[166]:


Sales_Price_df.groupby("Outlet_Type")["Item_Outlet_Sales"].mean().plot(kind="bar")
plt.xlabel="Outlet_Type"
plt.ylabel="Item_Outlet_Sales"
plt.show
plt.rcParams['figure.figsize'] = [5,2]


# #### BOX PLOT OF Outlet_Size,Outlet_Location_Type,Outlet_Type WITH Item_Outlet_Sales

# In[53]:


plt.figure(figsize = (10,9))

plt.subplot(311)
sns.boxplot(x='Outlet_Size', y='Item_Outlet_Sales', data=Sales_Price_df, palette="Set1")

plt.subplot(312)
sns.boxplot(x='Outlet_Location_Type', y='Item_Outlet_Sales', data=Sales_Price_df, palette="Set1")

plt.subplot(313)
sns.boxplot(x='Outlet_Type', y='Item_Outlet_Sales', data=Sales_Price_df, palette="Set1")

plt.subplots_adjust(wspace = 0.2, hspace = 0.4,top = 1.5)

plt.show()


# In[54]:


Sales_Price_df['Outlet_Size'] =Sales_Price_df['Outlet_Size'].replace(['Medium','Small'],['MS Combined Size','NWE Combined Metropolitan'])


# In[55]:


Sales_Price_df['Outlet_Location_Type'] =Sales_Price_df['Outlet_Location_Type'].replace(['Tier 1','Tier 3'],['1,2 Combined Tier','1,2 Combined Tier'])


# In[56]:


Sales_Price_df['Outlet_Type'] =Sales_Price_df['Outlet_Type'].replace(['Supermarket Type1','Supermarket Type2'],['1,2 Combined Type','1,2 Combined Type'])


# #### BOX PLOT OF Outlet_Identifier,Item_Type_Combined WITH Item_Outlet_Sales

# In[135]:



plt.figure(figsize = (14,9))

plt.subplot(211)
ax = sns.boxplot(x='Outlet_Identifier', y='Item_Outlet_Sales', data=Sales_Price_df, palette="Set1")
ax.set_title("Outlet_Identifier vs. Item_Outlet_Sales", fontsize=15)
ax.set_xlabel("", fontsize=12)
ax.set_ylabel("Item_Outlet_Sales", fontsize=12)

plt.subplot(212)
ax = sns.boxplot(x='Item_Type_Combined', y='Item_Outlet_Sales', data=Sales_Price_df, palette="Set1")
ax.set_title("Item_Type_Combined vs. Item_Outlet_Sales", fontsize=15)
ax.set_xlabel("", fontsize=12)
ax.set_ylabel("Item_Outlet_Sales", fontsize=12)

plt.subplots_adjust(hspace = 0.9, top = 0.9)
plt.setp(ax.get_xticklabels(), rotation=45)

plt.show()


# ### Statistical Hypothesis Tests for consideration of the effective columns

# In[58]:


mod=ols("Item_Outlet_Sales~Item_Fat_Content",data=Sales_Price_df).fit()
aov_table=sm.stats.anova_lm(mod,type=2)
print(aov_table)


# In[59]:


mod1=ols("Item_Outlet_Sales~Outlet_Size",data=Sales_Price_df).fit()
aov_table1=sm.stats.anova_lm(mod1,type=2)
print(aov_table1)


# In[60]:


mod3=ols("Item_Outlet_Sales~Item_Type",data=Sales_Price_df).fit()
aov_table2=sm.stats.anova_lm(mod3,type=2)
print(aov_table2)


# In[61]:


mod4=ols("Item_Outlet_Sales~Outlet_Identifier",data=Sales_Price_df).fit()
aov_table3=sm.stats.anova_lm(mod4,type=2)
print(aov_table3)


# In[62]:


mod5=ols("Item_Outlet_Sales~Outlet_Location_Type",data=Sales_Price_df).fit()
aov_table4=sm.stats.anova_lm(mod5,type=2)
print(aov_table4)


# In[63]:


mod6=ols("Item_Outlet_Sales~Outlet_Type",data=Sales_Price_df).fit()
aov_table5=sm.stats.anova_lm(mod6,type=2)
print(aov_table5)


# In[64]:


mod7=ols("Item_Outlet_Sales~Item_Identifier",data=Sales_Price_df).fit()
aov_table6=sm.stats.anova_lm(mod7,type=2)
print(aov_table6)


# In[65]:


cormat=Sales_Price_df.corr()
cormat


# In[67]:


plt.rcParams['figure.figsize'] = [10,5]
sns.heatmap(cormat,annot=True,vmax=+1,vmin=-1,annot_kws={"size":11.5})
plt.title("Correlation Matrix")
plt.show()


# In[57]:


Sales_Price_df.head()


# In[68]:


DF=Sales_Price_df.copy()


# In[69]:


DF.info()


# In[70]:


DF.drop(['Item_Identifier','Outlet_Identifier','Outlet_Establishment_Year','Item_Type'],axis=1,inplace=True)


# In[71]:


numerical_variables=DF.select_dtypes(include=np.number)


# In[72]:


numerical_variables.columns


# #### PAIR PLOT

# In[73]:


# Pairplot of numeric variables

# select the columns for the pairplot
columns= ["Item_Weight", "Item_Visibility", "Item_MRP", "Outlet_Years", "Item_Outlet_Sales"]

# draw the pairplot such that the diagonal should be density plot and the other graphs should be scatter plot
sns.pairplot(DF[columns], size=2, kind= "scatter", diag_kind="kde")

# display the plot
plt.show()


# ### 5.SGD REGRESSION

# In[83]:


import sklearn
from sklearn import tree
from sklearn import metrics
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor


# Seperating numerical and categorical variables

# In[74]:


numerical_variables=DF.select_dtypes(include=np.number)


# In[75]:


categorical_variables=DF.select_dtypes(include="object")


# In[182]:


dummy_enc_cat=pd.get_dummies(categorical_variables,drop_first=True)


# In[183]:


dummy_enc_cat.shape


# In[184]:


dummy_sales_df=pd.concat([numerical_variables,dummy_enc_cat],axis=1)


# In[185]:


dummy_sales_df.head()


# In[186]:


x=dummy_sales_df.drop(["Item_Outlet_Sales"],axis=1)


# In[187]:


y=dummy_sales_df[["Item_Outlet_Sales"]]


# In[188]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)
x_train=StandardScaler().fit_transform(x_train)
x_test=StandardScaler().fit_transform(x_test)
y_train=StandardScaler().fit_transform(y_train)
y_test=StandardScaler().fit_transform(y_test)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[189]:


from sklearn.linear_model import SGDRegressor
sgdr=SGDRegressor()
Linear_reg_SGD_model=sgdr.fit(x_train,y_train)
print(Linear_reg_SGD_model)


# In[190]:


score_SGD=sgdr.score(x_train,y_train)
print(score_SGD)


# In[191]:


Linear_reg_model_SGD_predict=Linear_reg_SGD_model.predict(x_test)


# In[200]:


Linear_reg_model_SGD_predict


# In[192]:


actual_Linear_reg_SGD_model=y_test


# In[193]:


SGD_msme=mean_squared_error(actual_Linear_reg_SGD_model,Linear_reg_model_SGD_predict)
SGD_rmse=np.sqrt(SGD_msme)
print(SGD_msme)
print(SGD_rmse)


# ### 6. Linear Regression (OLS)

# In[194]:


import statsmodels
import statsmodels.api as sm
from statsmodels.compat import lzip
import statsmodels.stats.api as sms
import statsmodels.formula.api as smf
from statsmodels.formula.api import ols
from statsmodels.tools.eval_measures import rmse
from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression


# In[195]:


# build a full model using OLS()
linreg_OLSmodel_full = sm.OLS(y_train, x_train).fit()

# print the summary output
print(linreg_OLSmodel_full.summary())


# In[196]:


Linear_reg_OLSmodel_predict=linreg_OLSmodel_full.predict(x_test)


# In[197]:


actual_Linear_reg_OLSmodel=y_test


# In[198]:



linreg_OLSmodel_full_rsquared = linreg_OLSmodel_full.rsquared

# calculate Adjusted R-Squared using rsquared_adj
linreg_OLSmodel_full_rsquared_adj = linreg_OLSmodel_full.rsquared_adj 


# In[199]:


print(linreg_OLSmodel_full_rsquared)
print(linreg_OLSmodel_full_rsquared_adj)


# ### 7.DECISION TREE & RANDOMFOREST

# In[152]:


#Decision Tree
clf = tree.DecisionTreeRegressor()
tf_clf = clf.fit(x_train, y_train)


# In[153]:


#Prediction
y_pred = tf_clf.predict(x_test)
print(y_pred)


# In[154]:


from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test, y_pred)
RMSE = np.sqrt(mse)
print(RMSE)


# In[155]:


rfr = RandomForestRegressor(n_estimators = 150) 
tf_rfr = rfr.fit(x_train, y_train)


# In[156]:


#Prediction
y_Pred = tf_rfr.predict(x_test)

#RMSE
from sklearn.metrics import mean_squared_error
RF_mse = mean_squared_error(y_test, y_Pred)
RF_RMSE = np.sqrt(RF_mse)
print(RF_RMSE) 


# ### 8. RESULT TABULATION

# In[326]:


# create the result table for all accuracy scores
# accuracy measures considered for model comparision are RMSE, R-squared value and Adjusted R-squared value
# create a list of column names
cols = ['Model',  'R-Squared']

# create a empty dataframe of the colums
# columns: specifies the columns to be selected
Result = pd.DataFrame(columns = cols)

# compile the required information
linreg_full_metrics = pd.Series({'Model': "Linear_reg_model_SGD ",
                     'R-Squared': score_SGD   
                   })

# append our result table using append()
# ignore_index=True: does not use the index labels
# python can only append a Series if ignore_index=True or if the Series has a name
Result = Result.append(linreg_full_metrics, ignore_index = True)

# print the result table
Result


# In[327]:


linreg_full_metrics = pd.Series({'Model': "Linear_reg_model_OLS",
                     'R-Squared':linreg_OLSmodel_full_rsquared 
                   })

# append our result table using append()
# ignore_index=True: does not use the index labels
# python can only append a Series if ignore_index=True or if the Series has a name
Result = Result.append(linreg_full_metrics, ignore_index = True)

# print the result table
Result

