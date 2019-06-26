# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 04:24:49 2019

@author: Shivangi Sharma
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data_set=pd.read_csv("sal.csv")

x=data_set.iloc[:,:-1].values
y=data_set.iloc[:,1].values

from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(xtrain,ytrain)

y_pred=regressor.predict(xtest)

'''plt.scatter(xtrain,ytrain,color='green')
plt.plot(xtrain,regressor.predict(xtrain),color='red')
plt.title('experience vs salary')
plt.xlabel('years of experience')
plt.ylabel('salary')'''

plt.scatter(xtest,ytest,color='blue')
plt.plot(xtrain,regressor.predict(xtrain),color='yellow')
plt.title('experience vs salary')
plt.xlabel('years of experience')
plt.ylabel('salary')