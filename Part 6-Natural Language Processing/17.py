import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset=pd.read_csv("resto.tsv",delimiter='\t')
import nltk
nltk.download('stopwords')

import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus=[]
for i in range(0,1000):
    review=re.sub('[^a-zA-Z]',' ',dataset['Review'][i])
    review=review.lower()
    review=review.split()
    ps=PorterStemmer()
    review=[ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)#for converting into string
    corpus.append(review)

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=1500)
x=cv.fit_transform(corpus).toarray()
y=dataset.iloc[:,1].values

from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.18,random_state=0)

from sklearn.naive_bayes import GaussianNB
classifier1=GaussianNB()
classifier1.fit(xtrain,ytrain)
ypred1=classifier1.predict(xtest)

from sklearn.metrics import confusion_matrix
cf1=confusion_matrix(ytest,ypred)

from sklearn.tree import DecisionTreeClassifier
classifier2=DecisionTreeClassifier()
classifier2.fit(xtrain,ytrain)
ypred2=classifier2.predict(xtest)
cf2=confusion_matrix(ytest,ypred2)





