#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os 
import numpy as np


# In[28]:


class preprocessor:
    
    #Initiliazing Features and Target variables
    def __init__(self, features, target):
        self.features = features
        self.target = target

        
    # Encoding categorical data using Label Encoding and OneHot encoding.
    def fit_transform(self):
        le = LabelEncoder()
        self.features[:, 2] = le.fit_transform(self.features[:, 2])
        ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
        self.features = np.array(ct.fit_transform(self.features))
        return self.features
   
    #Standard scaling is performed to transform train and test data
    def standard_scaling(self, features_train, features_test):
        sc = StandardScaler()
        features_train = sc.fit_transform(features_train)
        features_test = sc.transform(features_test)
        
        return features_train, features_test





