#!/usr/bin/env python
# coding: utf-8

# ![image.png](attachment:image.png)

# # Context
# 
# The data here provides insight on the body measurements and body fat percentage for 252 men. The aim of this study it develop a model to estimate the measurement of body fat percentage using body measurements. To test individuals for their body fat percentage can be extremely costly.

# In[3]:


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


# In[4]:


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

# In[5]:


df.shape


# In[6]:


df.info()
#no null values


# In[7]:


df.describe()


# In[8]:


df.isnull().sum()


# In[9]:


#no duplicates in our data
df.duplicated().sum()


# # Exploratory Data Analysis

# In[10]:


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


# In[11]:


# Checking Skewness
df.skew(axis=0).sort_values()


# ### Important Notes on Skewness:
# · If the skewness is between -0.5 and 0.5, the data are fairly symmetrical
# 
# · If the skewness is between -1 and — 0.5 or between 0.5 and 1, the data are moderately skewed
# 
# · If the skewness is less than -1 or greater than 1, the data are highly skewed

# In[12]:


#Checking to see highly skewed data
dfskew = df.skew(axis=0).sort_values().reset_index().rename(columns={'index':'Column', 0:'Skewness'})
dfskew[(dfskew['Skewness'] < -1) | (dfskew['Skewness'] > 1)]
# We can see areas height, weight, hip, and ankle are all heavily skewed


# In[13]:


plt.figure(figsize = (12,12))
sns.heatmap(df.corr(), annot = True, linewidths = 0.5, fmt="0.2f")
#Scale of heatmap:
# 1 means positively correlated
# -1 means inversely correlated


# #### Results:
# 
# The correlation matrix above is consisent with normal body fat distribution. For ex, bodyfat and abdomen have a 0.8 correlation. This makes sense because as abdomen circumference increases, you have a higher bodyfat and people generally tend to get fat on their abdomen first. 

# In[16]:


X = df.drop(['BodyFat','Density'], axis=1)
y = df['Density']


# In[17]:


X['Bmi'] = 703*X['Weight']/(X['Height']**2)
X.head()


# In[21]:


#Abdomen to Chest Ratio
X['ACratio'] = X['Abdomen']/X['Chest']
#Hip to Thigh Ratio
X['HTratio'] = X['Hip']/X['Thigh']

X.drop(['Weight','Height', 'Abdomen','Chest','Hip','Thigh'],axis=1,inplace=True)
X


# In[22]:


z = np.abs(stats.zscore(X))

#only keep rows in dataframe with all z-scores less than absolute value of 3
X_clean = X[(z<3).all(axis=1)]
y_clean = y[(z<3).all(axis=1)]

#find how many rows are left in the dataframe 
X_clean.shape


# In[23]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X_clean,y_clean,random_state=17)


# In[26]:


trans = PowerTransformer()
X_train = trans.fit_transform(X_train)
X_test = trans.transform(X_test)


# In[28]:


kernel = KernelRidge()
# lgbm = LGBMRegressor()
random = RandomForestRegressor()
linear = LinearRegression()
elastic = ElasticNet()
lasso  = Lasso()
ridge = Ridge()


clf = [linear,elastic,lasso,ridge,random,kernel]
hashmap={}


# In[29]:


from sklearn.metrics import mean_squared_error

#Defining compute function
def compute(model):
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    r2=r2_score(y_test,y_pred)
    rmse=np.sqrt(mean_squared_error(y_test,y_pred))
    hashmap[str(model)]=(r2,rmse)

for i in clf:
    compute(i)

hashmap


# In[30]:


score = pd.DataFrame(hashmap)
score = score.transpose()
score.columns=['R2_score','RMSE']
score = score.sort_values('R2_score',ascending=False)
score


# In[119]:


y_pred = linear.predict(X_test)
sns.scatterplot(y_test,y_pred, color='r')
plt.plot([1.02, 1.09], [1.02, 1.09], color = 'black',linestyle='--',linewidth=3)
plt.xlabel("Actual-->")
plt.ylabel("Predicted-->")
plt.title("Actual Vs Predicted")


# # Predicting Body Fat Percentage
# 
# Many body composition equations derive their measure of percent body fat from first determining body density. Once body density is determined, percent bodyfat (%BF) can be calculated using the Siri equation below :
# 
# % Body Fat = (495 / Body Density) - 450
# 

# In[41]:


def predict(values):
    density = linear.predict(values)
    fat = ((4.95/density[0]) - 4.5)*100
    print(f'Density: {density[0]} g/cc\nPercentage Body Fat: {fat} %\n')
    


# In[45]:


predict(X_test[26].reshape(1,-1))


# In[ ]:




