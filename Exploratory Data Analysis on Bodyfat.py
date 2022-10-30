#!/usr/bin/env python
# coding: utf-8

# ![image.png](attachment:image.png)

# # Context
# 
# The data here provides insight on the body measurements and body fat percentage for 252 men. The aim of this study it develop a model to estimate the measurement of body fat percentage using body measurements. To test individuals for their body fat percentage can be extremely costly.

# In[6]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from sklearn.preprocessing import PowerTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression , ElasticNet , Lasso , Ridge
from sklearn.metrics import r2_score
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.kernel_ridge import KernelRidge
plt.style.use('fivethirtyeight')
colors=['#ffcd94','#eac086','#ffad60','#ffe39f']
sns.set_palette(sns.color_palette(colors))


# In[7]:


df = pd.read_csv('bodyfat.csv')
df


# # Lets understand our data
# We have the following columns:
# - Density determined from underwater weighing
# - Percent body fat from Siri's (1956) equation
# - Age (years)
# - Weight (lbs)
# - Height (inches)
# - Neck circumference (cm)
# - Chest circumference (cm)
# - Abdomen 2 circumference (cm)
# - Hip circumference (cm)
# - Thigh circumference (cm)
# - Knee circumference (cm)
# - Ankle circumference (cm)
# - Biceps (extended) circumference (cm)
# - Forearm circumference (cm)
# - Wrist circumference (cm
# 
# 
# 

# In[8]:


df.shape


# In[9]:


df.info()
#no null values


# In[10]:


df.describe()


# In[11]:


df.isnull().sum()


# In[12]:


#no duplicates in our data
df.duplicated().sum()


# # Exploratory Data Analysis

# In[85]:


import warnings
import statsmodels.api as sm


warnings.filterwarnings('ignore')
fig,ax = plt.subplots(15,2,figsize=(30,90))
for i in enumerate(df.columns):
    sns.histplot(df[i[1]],ax=ax[i[0],0], kde=True)
    sns.boxplot(df[i[1]],ax=ax[i[0],1])
    
fig.tight_layout()
fig.subplots_adjust(top=0.95)
plt.suptitle("Visualizing Our Data",fontsize=50)


# In[88]:


# Checking Skewness
df.skew(axis=0).sort_values()


# # Important Notes on Skewness:
# · If the skewness is between -0.5 and 0.5, the data are fairly symmetrical
# 
# · If the skewness is between -1 and — 0.5 or between 0.5 and 1, the data are moderately skewed
# 
# · If the skewness is less than -1 or greater than 1, the data are highly skewed

# In[113]:


#Checking to see highly skewed data
dfskew = df.skew(axis=0).sort_values().reset_index().rename(columns={'index':'Column', 0:'Skewness'})
dfskew[(dfskew['Skewness'] < -1) | (dfskew['Skewness'] > 1)]
# We can see areas height, weight, hip, and ankle are all heavily skewed


# In[114]:


plt.figure(figsize = (12,12))
sns.heatmap(df.corr(), annot = True, linewidths = 0.5, fmt="0.2f")
#Scale of heatmap:
# 1 means positively correlated
# -1 means inversely correlated


# ## The correlation matrix above is consisent with normal body fat distribution. For ex, bodyfat and abdomen have a 0.8 correlation. This makes sense because as abdomen circumference increases, you have a higher bodyfat and people generally tend to get fat on their abdomen first. 

# In[ ]:




