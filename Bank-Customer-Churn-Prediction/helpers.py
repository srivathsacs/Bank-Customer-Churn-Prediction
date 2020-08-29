#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import os

def read_in_dataset(verbose=False):
    
    dicPath = os.getcwd()
    dicPathData = os.path.join(dicPath,'dataset','Churn_Modelling.csv')
    dataset = pd.read_csv(dicPathData)
    dataset.head()
    
    return dataset

