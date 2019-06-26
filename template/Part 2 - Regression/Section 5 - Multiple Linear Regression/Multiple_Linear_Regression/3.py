# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 10:55:46 2019

@author: Shivangi Sharma
"""


import pandas as pd
import numpy as np
data=pd.read_csv("50.csv")
x=data.iloc[:,:-1].values
y=data.iloc[:,-1].values


from sklearn.preprocessing import LabelEncoder,OneHotEncoder
encoder=LabelEncoder()
x[:,3]=encoder.fit_transform(x[:,3])
onehotencoder=OneHotEncoder(categorical_features=[3])
x=onehotencoder.fit_transform(x).toarray()
 
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)

y_pred=regressor.predict(x_test)



import statsmodels.formula.api as sm
x=np.append(arr=np.ones((50,1)).astype(int),values=x,axis=1)
x_opt=x[:,[0,1,2,3,4,5]]
regressor_OLS=sm.OLS(endog=y,exog=x_opt).fit()
regressor_OLS.summary()

