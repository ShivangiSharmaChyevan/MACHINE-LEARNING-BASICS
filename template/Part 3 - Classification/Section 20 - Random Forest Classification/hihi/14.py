# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 15:09:59 2019

@author: Shivangi Sharma
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data=pd.read_csv("soc.csv")
x=data.iloc[:,[2,3]].values
y=data.iloc[:,4].values

from sklearn.model_selection import train_test_split

xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.25,random_state=0)

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
xtrain=sc.fit_transform(xtrain)
xtest=sc.transform(xtest)

from sklearn.ensemble import RandomForestClassifier
classifier=RandomForestClassifier(n_estimators=10,random_state=0)
classifier.fit(xtrain,ytrain)
ypred=classifier.predict(xtest)

from sklearn.metrics import confusion_matrix
cf=confusion_matrix(ytest,ypred)

from matplotlib.colors import ListedColormap
xset,yset=xtrain,ytrain
x1,x2=np.meshgrid(np.arange(start=xset[:,0].min()-1,stop=xset[:,0].max()+1,step=0.01),
                  np.arange(start=xset[:,1].min()-1,stop=xset[:,1].max()+1,step=0.01))
plt.contourf(x1,x2,classifier.predict(np.array([x1.ravel(),x2.ravel()]).T).reshape(x1.shape),
             alpha=0.75,cmap=ListedColormap(("red","green")))

plt.xlim(x1.min(),x1.max())
plt.ylim(x2.min(),x2.max())

for i,j in enumerate(np.unique(yset)):
    plt.scatter(xset[yset==j,0],xset[yset==j,1],c=ListedColormap(("red","green"))(i),label=j)

plt.title("decision tree on training set")
plt.xlabel("AGE")
plt.ylabel("Salary")
plt.legend()
plt.show()

from sklearn.metrics import confusion_matrix
cf=confusion_matrix(ytest,ypred)

from matplotlib.colors import ListedColormap
xset,yset=xtest,ytest
x1,x2=np.meshgrid(np.arange(start=xset[:,0].min()-1,stop=xset[:,0].max()+1,step=0.01),
                  np.arange(start=xset[:,1].min()-1,stop=xset[:,1].max()+1,step=0.01))
plt.contourf(x1,x2,classifier.predict(np.array([x1.ravel(),x2.ravel()]).T).reshape(x1.shape),
             alpha=0.75,cmap=ListedColormap(("red","green")))

plt.xlim(x1.min(),x1.max())
plt.ylim(x2.min(),x2.max())

for i,j in enumerate(np.unique(yset)):
    plt.scatter(xset[yset==j,0],xset[yset==j,1],c=ListedColormap(("red","green"))(i),label=j)

plt.title("decision tree on test set")
plt.xlabel("AGE")
plt.ylabel("Salary")
plt.legend()
plt.show()


