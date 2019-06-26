import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data=pd.read_csv("mall.csv")
x=data.iloc[:,[3,4]].values

from sklearn.cluster import KMeans
wcss=[]
for i in range(1,11):
    kmeans=KMeans(n_clusters=i,init='k-means++',random_state=0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)
    
plt.plot(range(1,11),wcss)
plt.title("ELBOW METHOD")
plt.xlabel("no. of clusters")
plt.ylabel("wcss")
plt.show()

kmeans=KMeans(n_clusters=5,init='k-means++',random_state=0)
ypred=kmeans.fit_predict(x)

plt.scatter(x[ypred==0,0],x[ypred==0,1],s=100,c='red',label='cluster1')
plt.scatter(x[ypred==1,0],x[ypred==1,1],s=100,c='blue',label='cluster2')
plt.scatter(x[ypred==2,0],x[ypred==2,1],s=100,c='cyan',label='cluster3')
plt.scatter(x[ypred==3,0],x[ypred==3,1],s=100,c='brown',label='cluster4')
plt.scatter(x[ypred==4,0],x[ypred==4,1],s=100,c='yellow',label='cluster5')
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=200,c='black',label='centroids')
plt.title("clusters of customers")
plt.xlabel("annaul income")
plt.legend()
plt.ylabel("score")
plt.show()

