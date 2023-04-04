#!/usr/bin/env python
# coding: utf-8

# # ******     The Sparks Foundation     ******

# # Task 1 : Prediction Using Supervised Machine Learning

# # Predict the percentage of an student based on the number of study hours.

# # Objectives :
# 
# To predict the percentage of marks that a student is expected to score based upon the number of hours they studied by using Simple Linear Regression.
# 
# 
# 
# 
# 
# 

# # Author: Manoj Pralhad Patil

# In[2]:


# Importing Required Libraries 
import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt  
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns

#Libraries for model building

import statsmodels.api as sm 
from statsmodels.stats import diagnostic as diag 
from sklearn.model_selection import train_test_split


# In[7]:


# Reading data from remote link

df='https://raw.githubusercontent.com/AdiPersonalWorks/Random/master/student_scores%20-%20student_scores.csv'
data = pd.read_csv(df)


# In[8]:


data.head()


# In[9]:


data.tail()


# In[10]:


data.info()


# In[11]:


data.describe()


# # Checking for Outliers

# In[13]:


sns.boxplot(y='Scores',data=data)
plt.show()


# In[15]:


sns.boxplot(y='Hours',data=data)
plt.show()


# # As we can see above there is no outliers presents in out data set

# In[16]:


data.isna().sum()


# In[17]:


data.info()


# In[18]:


# Plotting 2D the distribution of scores
data.plot(x='Hours', y='Scores', style='o')  
plt.title('Hours vs Percentage')  
plt.xlabel('Hours Studied')  
plt.ylabel('Percentage Score')  
plt.show()


# # Interpretation
# 
# From the above graph, we can clearly see that there is a positive linear relation between the number of hours studied and percentage of score.
# 
# 
# 
# 

# In[19]:


#To find the correlation between the study hours and the percentage scores

data.corr(method='pearson')


# The correlation between the Hours and Scores is 0.9761 so the both variables are highly correlated with each other.

# # Data Partition

# In[29]:


X=data[['Hours']]
y=data[['Scores']]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(x,y,train_size=0.7,random_state=100)


# In[30]:


train = pd.concat([X_train,y_train], axis=1)
train.head()


# # Building a Model

# Hypothesis Testing
# 
# H0 = There is no relationship between the Hours & Scores
# 
# H1 = There is relationship between the Hours & Scores
# 
# Alpha =5% ( 0.05 )

# In[31]:


import statsmodels.formula.api as smf
model = smf.ols('Scores ~ Hours', data=train).fit()
model.summary()


# # Conclusion
# 
# We Rejected HO
# 
# See there is relationship between the Score & Hours studied
# 
# Accuracy of Model is 95.7%, So Model is fitting good
# 
# 
# Y = 1.4951 + 9.8717 * Hours

# In[32]:


train['fitted_value']=model.fittedvalues # prediction  or fitted value
train['residual']=model.resid  # error or residual

train.head()


# # Prediction on Test Data [unseen data]

# In[33]:


test=pd.concat([X_test,y_test],axis=1)
test.head()


# In[34]:


from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

regressor.fit(X_train, y_train)

line = regressor.coef_*X+regressor.intercept_

plt.scatter(X,y)

plt.plot(X, line)

plt.show()


# In[35]:


test['Predicted']=model.predict(test)
test


# # Evaluating the model

# # TEST

# In[36]:


from sklearn.metrics import r2_score
r2 = r2_score(test.Scores, test.Predicted)
print('R2 score for model Performance on Test', np.round(r2,2))


# # TRAIN
# 

# In[37]:


from sklearn.metrics import r2_score
r2 = r2_score(train.Scores, train.fitted_value)
print('R2 score for perfect model is', np.round(r2,2))


# In[38]:


from sklearn import metrics  
print('Mean Absolute Error:', 
      metrics.mean_absolute_error(test.Scores,test.Predicted))


# # Model is good fit 
# 
# # Performance on Test data is also Looking good fit

# # Let's check What will be predicted score if a student studies for 9.25 hrs/ day 

# In[41]:


live['Score_Prediction']=np.round(model.predict(live))
live


# # We can see that prediction score is 93% if a student studied 9.25hrs/day

# # !!!!!!!!!!......Thank You......!!!!!!!!!

# In[ ]:




