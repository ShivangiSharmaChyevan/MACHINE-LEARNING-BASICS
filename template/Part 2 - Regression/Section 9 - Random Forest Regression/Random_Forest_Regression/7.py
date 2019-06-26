# -*- coding: utf-8 -*-
"""
Created on Sat Jun 15 19:25:16 2019

@author: Shivangi Sharma
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data=pd.read_csv("los.csv")

x=data.iloc[:,1:2].values
y=data.iloc[:,2].values

from sklearn.ensemble import RandomForestRegressor
regressor=RandomForestRegressor(n_estimators=300,random_state=0)
regressor.fit(x,y)
y_pred=regressor.predict([[6.5]])

plt.scatter(x,y,color='red')
x_grid=np.arange(min(x),max(x),0.1)
x_grid=x_grid.reshape(len(x_grid),1)
plt.plot(x_grid,regressor.predict(x_grid),color='black')


