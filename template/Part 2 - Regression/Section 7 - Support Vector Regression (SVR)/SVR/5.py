# -*- coding: utf-8 -*-
"""
Created on Sat Jun 15 13:36:00 2019

@author: Shivangi Sharma
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data=pd.read_csv("Position_Salaries.csv")

x=data.iloc[:,1:2].values
y=data.iloc[:,2:3].values

from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
sc_y=StandardScaler()
x=sc_x.fit_transform(x)
y=sc_y.fit_transform(y)

from sklearn.svm import SVR
regressor=SVR(kernel='rbf')

regressor.fit(x,y)

y_pred=sc_y.inverse_transform(regressor.predict(sc_x.fit_transform(np.array([[6.5]]))))

plt.scatter(x,y,color='red')
x_grid=np.arange(min(x),max(x),0.1)
x_grid=x_grid.reshape(len(x_grid),1)
plt.plot(x_grid,regressor.predict(x_grid),color='blue')