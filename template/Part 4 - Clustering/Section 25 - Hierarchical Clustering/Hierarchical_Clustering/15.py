# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 18:06:48 2019

@author: Shivangi Sharma
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data=pd.read_csv("mall.csv")
x=data.iloc[:,[3,4]].values

import scipy.cluster.hierarchy as sch
dendogram=sch.dendrogram(sch.linkage(x,method='ward'))
plt.title('DENDOGRAM')
plt.xlabel('CUSTOMERS')
plt.ylabel('Euclidean distance')
plt.show()

from sklearn.cluster import AgglomerativeClustering
hc=AgglomerativeClustering(n_clusters=5,affinity="euclidean",linkage="ward")
y_pred=hc.fit_predict(x)

plt.scatter(x[y_pred==0,0],x[y_pred==0,1],s=100,c='red',label='cluster1')
plt.scatter(x[y_pred==1,0],x[y_pred==1,1],s=100,c='green',label='cluster2')
plt.scatter(x[y_pred==2,0],x[y_pred==2,1],s=100,c='blue',label='cluster3')
plt.scatter(x[y_pred==3,0],x[y_pred==3,1],s=100,c='magenta',label='cluster4')
plt.scatter(x[y_pred==4,0],x[y_pred==4,1],s=100,c='cyan',label='cluster5')

plt.legend()
plt.xlabel("SALRY")
plt.ylabel("EXPENDITURE")
plt.title("clustering of customers")
