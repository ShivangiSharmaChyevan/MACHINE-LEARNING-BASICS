# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 21:59:07 2019

@author: Shivangi Sharma
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data=pd.read_csv("market.csv",header=None)

transactions=[]

for i in range(0,7501):
    transactions.append([str(data.values[i,j])for j in range(0,20)])

from apyori import apriori
rules=apriori(transactions,min_support=0.003,min_confidence=0.2,min_lift=3,min_length=2)

result=list(rules)
    
    