#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import confusion_matrix, accuracy_score
import Preprocessing
from Preprocessing import preprocessor
import matplotlib.pyplot as plt

# In[ ]:


class Model:
    
    # Initializing the Train-Test Split Data variables for the Class.
    def __init__(self, features_train, features_test, target_train, target_test):
        self.features_train = features_train
        self.features_test = features_test
        self.target_train = target_train
        self.target_test = target_test
        
    # Building an ANN model using the best Hyper parameters from GridSearch with Dropout as Regularization
    def model(self):
        ann = tf.keras.models.Sequential()
        ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
        ann.add(tf.keras.layers.Dropout(0.2))
        ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
        ann.add(tf.keras.layers.Dropout(0.2))
        ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
        
        ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    
        ann.fit(self.features_train, self.target_train, batch_size = 32, epochs = 100)
    
        return ann
    
    def model_performance(self, classifier):
        losses_lstm = classifier.history.history['loss']
        plt.figure(figsize=(12,4))
        plt.title(label = 'Model Performance given by Loss Function V/S Epochs')
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.xticks(np.arange(0,21,1))
        plt.plot(range(len(losses_lstm)),losses_lstm)
        
        
    #Predicting on the Test Data 
    def predict(self, ann):
        y_pred = ann.predict(self.features_test)
        y_pred = (y_pred > 0.5)
        print()
        
        print('********************************')
        print()
        test_results = np.concatenate((y_pred.reshape(len(y_pred),1),self.target_test.reshape(len(self.target_test),1)),1)
        print('Predicting on Test results\n', test_results)
        print()
        
        
        #Performing Confusion Matrix and Accuracy Metrics for checking the Model Performance
        print('********************************')
        print()
        cm = confusion_matrix(self.target_test, y_pred)
        print('Confusion Matrix:\n', cm)
        print()
        
        print('********************************')
        print()
        print("Model Accuracy:", (accuracy_score(self.target_test, y_pred) * 100))
        
        
        
        