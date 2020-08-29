#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from sklearn.metrics import confusion_matrix, accuracy_score
import Preprocessing
from Preprocessing import preprocessor
from helpers import read_in_dataset
from Model_Building import Model
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential
from ipynb.fs.full.GridSearch import grid_search


if __name__ == '__main__':
    
    #Reading Dataset
    dataset = read_in_dataset()
    
    #Seperating features and target values 
    features = dataset.iloc[:, 3:-1].values
    target = dataset.iloc[:, -1].values
    
    #Performing Data Preprocessing and transforming features
    p = preprocessor(features, target)
    transformed_features = p.fit_transform()
    
    #Splitting dataset into Train and Test Dataset
    features_train, features_test, target_train, target_test = train_test_split(transformed_features, target, 
                                                                                test_size = 0.2, random_state = 0)

    #Performing Standard Scaling
    train_features, test_features = p.standard_scaling(features_train, features_test)
    
    #Hyperparameter Tuning using GridSerach to find the best parameters for our model
    #best_parameters = grid_search(train_features, target_train)
    #print("The best parameters are:", best_parameters)
    
    #Building Artificial Neural Networks Model and testing on the test data, to check model performance
    m = Model(train_features, test_features, target_train, target_test)
    ann_model = m.model()
    m.predict(ann_model)
    m.model_performance(ann_model)
    
    


# In[ ]:





# In[ ]:




