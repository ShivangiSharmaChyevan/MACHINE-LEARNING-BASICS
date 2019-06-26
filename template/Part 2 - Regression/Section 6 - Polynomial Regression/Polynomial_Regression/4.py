import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data=pd.read_csv("pos3.csv")

x=data.iloc[:,1:2].values
y=data.iloc[:,2].values

from sklearn.linear_model import LinearRegression
lin_reg=LinearRegression()
lin_reg.fit(x,y)



from sklearn.preprocessing import PolynomialFeatures
poly_reg=PolynomialFeatures(degree=4)
x_poly=poly_reg.fit_transform(x)
lin_reg2=LinearRegression()
lin_reg2.fit(x_poly,y)

#plotting graphs
plt.scatter(x,y,color='red')
plt.plot(x,lin_reg.predict(x),color='blue')
x_grid=np.arange(min(x),max(x),0.1)
x_grid=x_grid.reshape(len(x_grid),1)
plt.plot(x_grid,lin_reg2.predict(poly_reg.fit_transform(x_grid)),color='green')

lin_reg.predict([[6.5]])
p=poly_reg.fit_transform(6.5)
lin_reg2.predict([[p]])