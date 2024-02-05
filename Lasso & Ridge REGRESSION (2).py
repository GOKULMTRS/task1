#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
df=pd.read_csv('Melbourne_housing.csv')
df


# In[ ]:


newdf=df.drop(columns=['Address', 'Date', 'Postcode', 'YearBuilt', 'Lattitude', 'Longtitude'])
newdf


# In[ ]:


df=newdf.dropna(inplace=True)
newdf


# In[ ]:


print(newdf.to_string())


# In[ ]:


newdf.info()


# In[ ]:


x=newdf.drop(columns=['Price'])
y=newdf['Price']
x


# In[ ]:


x_=pd.get_dummies(x,dtype='int')
x_


# In[ ]:


from sklearn.model_selection import train_test_split as tts
x_train,x_test,y_train,y_test=tts(x_,y,test_size=0.3)


# In[ ]:


from sklearn.linear_model import LinearRegression as linr
reg1=linr()
reg1.fit(x_train,y_train)


# In[ ]:


reg1.score(x_train,y_train)


# In[ ]:


reg1.score(x_test,y_test)           # THE NEGATIVE VALUE OF THE SCORE INDICATES THAT THE MODEL IS OVERFIT WHICH INDICATES THE MODEL HAS OVER ACCURACY THAN NORMAL (0.6 - 0.8) & UNDER FIT (0.2 - 0.6)


# In[ ]:


# TO AVOID THIS OVERFIT WE CAN USE LASSO & RIGID (L1 & L2) REGRESSION, BECAUSE IN LINEAR REGRESSION AS THE NUMBER OF COLUMN INPUTS (LIKE 629 ABOVE) THE GRAPH COUNTS INCREASES AND ACCURACY DECREASES.


# In[ ]:


from sklearn.linear_model import Lasso
reg2=Lasso(alpha=50,max_iter=100,tol=0.1)
reg2.fit(x_train,y_train)


# In[ ]:


reg2.score(x_train,y_train)              # IN LASSO REGRESSION THE ACCURACY FAULT WILL NOT BE HAPPENING BEACUSE IT WILL COMPARE THE BOTH LASSO AND LINEAR REGRESSION OUTPUTS.


# In[ ]:


reg2.score(x_test,y_test)


# In[ ]:


from sklearn.linear_model import Ridge
reg3=Ridge(alpha=50,max_iter=100,tol=0.1)  # IN RIDGE REGRESSION THE ACCURACY FAULT WILL NOT BE HAPPENING BEACUSE IT WILL COMPARE THE BOTH RIDGE AND LINEAR REGRESSION OUTPUTS.
reg3.fit(x_train,y_train)


# In[ ]:


reg3.score(x_train,y_train)  


# In[ ]:


reg3.score(x_test,y_test)


# In[6]:


df1=pd.read_csv('Top Indian Places to Visit.csv')
df1


# In[7]:


df2=df1.drop(columns=['Unnamed: 0','Zone','State','City','Name','Establishment Year','time needed to visit in hrs','Airport with 50km Radius','Weekly Off','Significance','DSLR Allowed','Best Time to visit'])
df2


# In[8]:


df1 = df2.drop(df2[df2['Entrance Fee in INR'] == 0].index)
df1


# In[9]:


df1.nunique()


# In[10]:


x1=df1.drop(columns=['Entrance Fee in INR'])
y1=df1['Entrance Fee in INR']
x1
y1


# In[11]:


x__=pd.get_dummies(x1,dtype='int')
x__


# In[12]:


x1=x__
x1


# In[13]:


from sklearn.model_selection import train_test_split as tts
x1_train,x1_test,y1_train,y1_test=tts(x1,y1,test_size=0.3)


# In[14]:


from sklearn.linear_model import LinearRegression as linr
reg4=linr()
reg4.fit(x1_train,y1_train)


# In[15]:


from sklearn.linear_model import Ridge
reg5=Ridge(alpha=50,max_iter=100,tol=0.1)
reg5.fit(x1_train,y1_train)


# In[16]:


from sklearn.linear_model import Lasso
reg6=Lasso(alpha=50,max_iter=100,tol=0.1)
reg6.fit(x1_train,y1_train)


# In[20]:


reg4.score(x1_train,y1_train)                          # IN THIS CSV FILE LINEAR REGRESSION GIVES HIGHEST ACCURACY


# In[21]:


reg4.score(x1_test,y1_test)


# In[22]:


reg5.score(x1_train,y1_train)


# In[23]:


reg5.score(x1_test,y1_test)


# In[24]:


reg6.score(x1_test,y1_test)


# In[25]:


reg6.score(x1_train,y1_train)


# In[26]:


import pandas as pd
daf=pd.read_csv('apple_quality.csv')
daf


# In[27]:


daf.nunique()


# In[28]:


daf.dropna(inplace=True)
daf


# In[29]:


daf_=daf.drop(columns=['A_id'])
daf_


# In[30]:


daf=daf_
daf


# In[31]:


x3=daf.drop(columns=['Quality'])
y3=daf['Quality']
y3


# In[32]:


from sklearn.model_selection import train_test_split as tts
x3_train,x3_test,y3_train,y3_test=tts(x3,y3,test_size=0.3)


# In[33]:


from sklearn.tree import DecisionTreeClassifier as dtc
mod1=dtc()
mod1.fit(x3_train,y3_train)


# In[34]:


mod1.score(x3_train,y3_train)


# In[35]:


mod1.score(x3_test,y3_test)            # 80% PERCENTAGE ACCURACY.


# In[36]:


ans = mod1.predict([[0.059386 , -1.067408 , -3.714549 , 0.473052 , 1.697986 , 2.244055 , 0.137784369]])
ans


# In[37]:


import pandas as pd
df__=pd.read_csv('student_exam_data.csv')
df__


# In[38]:


df__.dropna(inplace=True)


# In[39]:


df3=df__
df3


# In[40]:


x4=df3.drop(columns=['Pass/Fail'])
y4=df3['Pass/Fail']
y4


# In[41]:


from sklearn.model_selection import train_test_split as tts
x4_train,x4_test,y4_train,y4_test=tts(x4,y4,test_size=0.3)


# In[42]:


from sklearn.linear_model import LogisticRegression as logr
mod2=logr()
mod2.fit(x4_train,y4_train)


# In[43]:


mod2.score(x4_train,y4_train)


# In[44]:


mod2.score(x4_test,y4_test)


# In[45]:


from sklearn.tree import DecisionTreeClassifier as dtc
mod3=dtc()
mod3.fit(x4_train,y4_train)


# In[46]:


mod3.score(x4_train,y4_train)


# In[47]:


mod3.score(x4_test,y4_test)


# In[48]:


xy = mod3.predict([[4,45]])
xy


# In[49]:


import pandas as pd
df4=pd.read_csv('NY-House-Dataset.csv')
df4


# In[ ]:




