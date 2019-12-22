# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 21:51:50 2019

@author: INTENTODEMUSICO
"""
#!/usr/bin/env python
# coding: utf-8
# ## Librerías
from __future__ import absolute_import, division, print_function, unicode_literals
import os
import tensorflow as tf
print("TensorFlow version: {}".format(tf.__version__))
print("Eager execution: {}".format(tf.executing_eagerly()))
# Dataset
train_dataset_url = "https://raw.githubusercontent.com/intentodemusico/Stegonomono/master/ANN/Dataset/train_70000.csv"
test_dataset_url = "https://raw.githubusercontent.com/intentodemusico/Stegonomono/master/ANN/Dataset/test_70000.csv"
import pandas as pd

#%% Importing the dataset
trainDataset = pd.read_csv(train_dataset_url)
X_Train = trainDataset.iloc[:, 0:8].values
Y_Train = trainDataset.iloc[:, 7].values

testDataset = pd.read_csv(test_dataset_url)
X_Test = testDataset.iloc[:, 0:8].values
Y_Test = testDataset.iloc[:, 7].values

#%% Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_Train = sc.fit_transform(X_Train)
X_Test = sc.transform(X_Test)


model = tf.keras.Sequential([
  tf.keras.layers.Dense(8, activation=tf.nn.relu, input_shape=(8,)),  # input shape required
  tf.keras.layers.Dense(4, activation=tf.nn.relu),
  tf.keras.layers.Dense(2, activation=tf.nn.relu),
  tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)
])

model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

model.fit(X_Train, Y_Train, batch_size = 64, nb_epoch = 250)

print(model)
model.evaluate(X_Train, Y_Train)
#%% Predicting the Test set results
y_pred = model.predict(X_Test)
print(y_pred)
y_pred = (y_pred > 0.5)
print(y_pred)

#%% Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_Test, y_pred)
