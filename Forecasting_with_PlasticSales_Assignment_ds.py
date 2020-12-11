#!/usr/bin/env python
# coding: utf-8

# # Importing the required dataset

# In[1]:


import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt
import seaborn as sns


# # Loading the dataset

# In[2]:


month=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
plastic=pd.read_csv(r"C:\Users\Binita Mandal\Desktop\finity\forecasting\PlasticSales.csv")


# In[3]:


# Head of the data
plastic.head()


# In[4]:


# Tails of the data
plastic.tail()


# In[5]:


# Shape
plastic.shape


# In[6]:


# Size
plastic.size


# In[7]:


#Columns
plastic.columns


# In[8]:


# PLotting the data
plastic.Sales.plot()


# In[9]:


# Lets see what is the dataset about
plastic


# In[10]:


plastic["Date"] = pd.to_datetime(plastic.Month,format="%b-%y")
#look for c standard format codes

# Extracting Day, weekday name, month name, year from the Date column using 
# Date functions from pandas 

plastic["month"] = plastic.Date.dt.strftime("%b") # month extraction
plastic["year"] = plastic.Date.dt.strftime("%Y") # year extraction

#plastic["Day"] = plastic.Date.dt.strftime("%d") # Day extraction
#plastic["wkday"] = plastic.Date.dt.strftime("%A") # weekday extraction


# ### Heatmap

# In[11]:


plt.figure(figsize=(12,8))
heatmap_y_month = pd.pivot_table(data=plastic,values="Sales",index="year",columns="month",aggfunc="mean",fill_value=0)
sns.heatmap(heatmap_y_month,annot=True,fmt="g") #fmt is format of the grid values


# ##### Boxplot for ever

# In[12]:


plt.figure(figsize=(8,6))
plt.subplot(211)
sns.boxplot(x="month",y="Sales",data=plastic)
plt.subplot(212)
sns.boxplot(x="year",y="Sales",data=plastic)


# ### Preparing Dummy variables

# In[13]:


month =['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'] 
import numpy as np
p = plastic["Month"][0]
p[0:3]
plastic['months']= 0

for i in range(60):
    p = plastic["Month"][i]
    plastic['months'][i]= p[0:3]
    
month_dummies = pd.DataFrame(pd.get_dummies(plastic['months']))
plastic1 = pd.concat([plastic,month_dummies],axis = 1)

t=np.arange(1,61)
plastic1['t']=t
t_square=plastic1['t']*plastic1['t']
plastic1['t_square']=t_square

log_Sales=np.log(plastic1['Sales'])

plastic1['log_Sales']=log_Sales
#plastic1["t_squared"] = plastic1["t"]*plastic1["t"]
#plastict1.columns
#plastic1["log_Sales"] = np.log(plastic1["Sales "])
#plastic1.rename(columns={"Sales ": 'Sales'}, inplace=True)
#plastic1.Sales.plot()


# ### Lineplot

# In[14]:


plt.figure(figsize=(12,3))
sns.lineplot(x="year",y="Sales",data=plastic)


# In[15]:


decompose_ts_add = seasonal_decompose(plastic.Sales,period=12)
decompose_ts_add.plot()
plt.show()


# # Splitting data

# In[16]:


Train=plastic1.head(48)
Test=plastic1.tail(12)


# In[17]:


plastic1.Sales.plot()


# In[18]:


#Linear Model
import statsmodels.formula.api as smf 

linear_model = smf.ols('Sales~t',data=Train).fit()
pred_linear =  pd.Series(linear_model.predict(pd.DataFrame(Test['t'])))
rmse_linear = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(pred_linear))**2))
rmse_linear


# In[19]:


#Exponential

Exp = smf.ols('log_Sales~t',data=Train).fit()
pred_Exp = pd.Series(Exp.predict(pd.DataFrame(Test['t'])))
rmse_Exp = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(np.exp(pred_Exp)))**2))
rmse_Exp


# In[20]:


#Quadratic 

Quad = smf.ols('Sales~t+t_square',data=Train).fit()
pred_Quad = pd.Series(Quad.predict(Test[["t","t_square"]]))
rmse_Quad = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(pred_Quad))**2))
rmse_Quad


# In[21]:


#Additive seasonality 

add_sea = smf.ols('Sales~Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data=Train).fit()
pred_add_sea = pd.Series(add_sea.predict(Test[['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov']]))
rmse_add_sea = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(pred_add_sea))**2))
rmse_add_sea


# In[22]:


#Additive Seasonality Quadratic 

add_sea_Quad = smf.ols('Sales~t+t_square+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data=Train).fit()
pred_add_sea_quad = pd.Series(add_sea_Quad.predict(Test[['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','t','t_square']]))
rmse_add_sea_quad = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(pred_add_sea_quad))**2))
rmse_add_sea_quad


# In[23]:


##Multiplicative Seasonality

Mul_sea = smf.ols('log_Sales~Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data = Train).fit()
pred_Mult_sea = pd.Series(Mul_sea.predict(Test))
rmse_Mult_sea = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(np.exp(pred_Mult_sea)))**2))
rmse_Mult_sea


# In[24]:


#Multiplicative Additive Seasonality 

Mul_Add_sea = smf.ols('log_Sales~t+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data = Train).fit()
pred_Mult_add_sea = pd.Series(Mul_Add_sea.predict(Test))
rmse_Mult_add_sea = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(np.exp(pred_Mult_add_sea)))**2))
rmse_Mult_add_sea 


# In[25]:


#Compare the results 

data = {"MODEL":pd.Series(["rmse_linear","rmse_Exp","rmse_Quad","rmse_add_sea","rmse_add_sea_quad","rmse_Mult_sea","rmse_Mult_add_sea"]),"RMSE_Values":pd.Series([rmse_linear,rmse_Exp,rmse_Quad,rmse_add_sea,rmse_add_sea_quad,rmse_Mult_sea,rmse_Mult_add_sea])}
table_rmse=pd.DataFrame(data)
table_rmse.sort_values(['RMSE_Values'])


# #### Multiplicative Additive Seasonality is Best fit

# # Predicting the new time period

# In[26]:


predict_data = pd.read_excel(r"C:\Users\Binita Mandal\Desktop\finity\forecasting\New_PlasticSales.xlsx")


# In[27]:


predict_data


# In[28]:


#Build the model on entire data set
model_full = smf.ols('Sales~t+t_square',data=plastic1).fit()
pred_new  = pd.Series(model_full.predict(predict_data))
pred_new


# In[29]:


predict_data["forecasted_Sales"] = pd.Series(pred_new)
predict_data


# ### Here I have got the forecasted value for next 11 months along with t_square values

# In[ ]:




