#!/usr/bin/env python
# coding: utf-8

# In[117]:


import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler, PowerTransformer
import matplotlib.pyplot as plt
import numpy as np
import pickle
from sklearn.metrics import r2_score


# In[ ]:





# In[ ]:





# In[118]:


df = pd.read_excel("regression_data.xls")


# Limpieza y mover cosa de data

# In[119]:


df


# In[120]:


df = df.drop(columns=["id", "date"])


# In[121]:


df


# In[122]:


df.info()


# In[123]:


df["bathrooms"] = df["bathrooms"].astype(int)


# In[124]:


df["floors"] = df["floors"].astype(int)


# In[125]:


df.info()


# In[126]:


df.describe()


# In[127]:


plt.figure(figsize=(30, 30))
sns.histplot(data = df, x = "price")


# In[ ]:





# In[ ]:





# In[ ]:





# In[128]:


plt.figure(figsize=(15, 15))
sns.heatmap(df.corr(), annot=True)


# In[129]:


df


# In[ ]:





# In[ ]:





# In[130]:


price_per_zip = {}

zips = df["zipcode"]
prices = df["price"]
for i in range(len(zips)):
    if zips[i] in price_per_zip.keys():
        price_per_zip[zips[i]].append(prices[i])
    else:
        price_per_zip[zips[i]] = []
        price_per_zip[zips[i]].append(prices[i])


# In[131]:


mean_price_per_zip = {}
for key in price_per_zip.keys():
    mean_price_per_zip[key] = np.mean(price_per_zip[key])


# In[132]:


median_price_list = []
keys = list(mean_price_per_zip.keys())
for i in range(len(zips)):
    median_price_list.append(mean_price_per_zip[zips[i]])


# In[133]:


df["mean_price_per_zipcode"] = median_price_list


# In[134]:


df


# In[135]:


#linear_model = df.drop(columns=["sqft_lot15", "long", "lat", "zipcode", "yr_renovated", "yr_built", "condition", "sqft_lot"])
linear_model = df


# In[136]:


linear_model = linear_model.drop(columns=["mean_price_per_zipcode"])


# In[137]:


linear_model


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# Desde aqui todo es escalar data

# In[138]:


from sklearn.model_selection import train_test_split


# In[139]:


X = df.drop(columns=["price"])
Y = df["price"]
X_lineal = linear_model.drop(columns=["price"])
Y_lineal = linear_model["price"]


# In[140]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)


# In[141]:


X_train_lineal, X_test_lineal, Y_train_lineal, Y_test_lineal = train_test_split(X_lineal, Y_lineal, test_size=0.3, random_state=42)


# In[142]:


pt = PowerTransformer()


# In[143]:


pt.fit(X_train_lineal)


# In[144]:


with open("scaler.pkl", "wb") as f:
    pickle.dump(pt, f)


# In[145]:


X_train_scaled = pt.transform(X_train_lineal)
X_test_scaled = pt.transform(X_test_lineal)


# In[146]:


Y_train_scaled = np.log(Y_train_lineal)


# In[147]:


X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_lineal.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_lineal.columns)


# In[148]:


X_train_scaled


# In[149]:


X_test_scaled


# Cosa de modelos

# Linear Regression

# In[150]:


from sklearn.linear_model import LinearRegression


# In[151]:


lr = LinearRegression()


# In[152]:


lr.fit(X_train_scaled, Y_train_scaled)


# In[153]:


lineal_pred = lr.predict(X_test_scaled)


# In[154]:


lineal_pred = np.exp(lineal_pred)


# In[155]:


lineal_pred


# In[156]:


r2_score(Y_test_lineal, lineal_pred)


# In[157]:


plt.figure(figsize=(20, 20))
sns.histplot([lineal_pred, Y_test_lineal])


# In[158]:


res_linear = lineal_pred - Y_test_lineal


# In[159]:


res_linear


# In[160]:


plt.figure(figsize=(30, 30))
plt.axhline()
plt.axhline(y = 100000)
sns.scatterplot(x = Y_test_lineal, y = res_linear)


# In[161]:


plt.figure(figsize=(15, 15))
plt.axvline(x = 100000, color = "red")
plt.axvline(x = -100000, color = "red")
sns.histplot(x = res_linear)


# In[162]:


with open("linear_regresion_model.pkl", "wb") as f:
    pickle.dump(lr, f)

#In[]


# Gradient Boosting

# In[180]:


from sklearn.ensemble import GradientBoostingRegressor


# In[181]:


gb = GradientBoostingRegressor(n_estimators=1000)


# In[182]:


gb.fit(X_train, Y_train)


# In[183]:


predict_gb = gb.predict(X_test)


# In[184]:


gb.score(X_test, Y_test)


# In[185]:


predict_gb


# In[186]:


errosr = Y_test - predict_gb


# In[187]:


errosr


# In[188]:


plt.figure(figsize=(20, 20))
sns.histplot(x = res_linear)
sns.histplot(x = errosr, color = "green")


# In[189]:


plt.figure(figsize=(30, 30))
plt.axhline()
plt.axhline(y = 100000)
sns.scatterplot(x = Y_test, y = errosr)


# In[190]:


with open("gradient_boosting_model.pkl", "wb") as f:
    pickle.dump(gb, f)


# %%

# %%

# %%

# %%
