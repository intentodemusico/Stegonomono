# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 21:51:50 2019

@author: INTENTODEMUSICO
"""

#!/usr/bin/env python
# coding: utf-8

# ## Librerías

# In[1]:


from __future__ import absolute_import, division, print_function, unicode_literals

import os
import matplotlib.pyplot as plt


# Descomentar para correr con gugol colab

# In[2]:


#%tensorflow_version 2.x


# In[3]:


from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())


# In[4]:


import tensorflow as tf


# In[5]:


print("TensorFlow version: {}".format(tf.__version__))
print("Eager execution: {}".format(tf.executing_eagerly()))


# Dataset

# In[6]:


train_dataset_url = "https://raw.githubusercontent.com/intentodemusico/Stegonomono/master/ANN/Dataset/train_70000.csv"

train_dataset_fp = tf.keras.utils.get_file(fname=os.path.basename(train_dataset_url),
                                           origin=train_dataset_url)
print("Local copy of the dataset file: {}".format(train_dataset_fp))


# In[7]:


# column order in CSV file
column_names = ['Kurtosis', 'Skewness', 'Std', 'Range', 'Median', 'Geometric_Mean', 'Mobility', 'Complexity','Steganography']

feature_names = column_names[:-1]
label_name = column_names[-1]

print("Features: {}".format(feature_names))
print("Label: {}".format(label_name))


# In[8]:


class_names = [0,1]


# In[9]:


batch_size = 64

train_dataset = tf.data.experimental.make_csv_dataset(
    train_dataset_fp,
    batch_size,
    column_names=column_names,
    label_name=label_name,
    num_epochs=250)


# In[10]:


features, labels= next(iter(train_dataset))
features['Range']=tf.dtypes.cast(features['Range'], tf.float32)
labels=tf.dtypes.cast(labels, tf.float32)


# In[11]:


print(labels)


# In[12]:


def pack_features_vector(features, labels):
  """Pack the features into a single array."""
  features['Range']=tf.dtypes.cast(features['Range'], tf.float32)#Machetazo supremo
  print(features)
  features = tf.stack(list(features.values()), axis=1)
  return features, labels


# In[13]:


train_dataset = train_dataset.map(pack_features_vector)


# In[14]:


features, labels = next(iter(train_dataset))

#print(features[:5])
#print("\n" ,labels)


# In[15]:


labels=tf.dtypes.cast(labels, tf.float32)#Machetazo supremo
#labels=tf.reshape(labels,(1,64))

print(labels)


# ## Modelling

# In[16]:


model = tf.keras.Sequential([
  tf.keras.layers.Dense(8, activation=tf.nn.relu, input_shape=(8,)),  # input shape required
  tf.keras.layers.Dense(4, activation=tf.nn.relu),
  tf.keras.layers.Dense(2, activation=tf.nn.relu),
  tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)
])


# In[17]:


#predictions = model(features)
#predictions[:5]


# In[18]:


#tf.nn.softmax(predictions[:5])


# In[19]:


#print("Prediction: {}".format(tf.argmax(predictions, axis=1)))
#print("    Labels: {}".format(labels))


# ## Training

# In[20]:



#model.compile(optimizer='adam',
 #             loss='binary_crossentropy',
  #            metrics=['accuracy'])


# In[21]:


#loss_object = tf.keras.losses.binary_crossentropy(from_logits=True)


# ## Aquí podría hacer el reshape para evitar el error...

# In[22]:


def loss(model, x, y):
  #y_ = tf.reshape(model(x),(64,)) #Wtf acá
  y_=model(x)#print(y)
  return tf.keras.losses.binary_crossentropy(from_logits=True,y_true=y, y_pred=y_)

#features=tf.reshape(features,(1,64))
labels=tf.reshape(labels,(64,1))
l = loss(model, features, labels)
print("Loss test: {}".format(l))


# In[23]:


def grad(model, inputs, targets):
  targets=tf.reshape(targets,(int(targets.get_shape().as_list()[0]),1))
  with tf.GradientTape() as tape:
    loss_value = loss(model, inputs, targets)
  return loss_value, tape.gradient(loss_value, model.trainable_variables)


# In[24]:


optimizer = tf.keras.optimizers.Adam(learning_rate=0.01) #Optimization function
#print(labels)


# In[25]:


## Note: Rerunning this cell uses the same model variables

# Keep results for plotting
train_loss_results = []
train_accuracy_results = []

num_epochs = 250


for epoch in range(num_epochs):
  epoch_loss_avg = tf.keras.metrics.Mean()
  epoch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

  # Training loop - using batches of 64
  for x, y in train_dataset:
    # Optimize the model
    #print("x:",x,"y:",y,"\n")
    loss_value, grads = grad(model, x, y)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    # Track progress
    epoch_loss_avg(loss_value)  # Add current batch loss
    # Compare predicted label to actual label
    y=tf.reshape(y,(int(y.get_shape().as_list()[0]),1))
    epoch_accuracy(y, model(x))
    

  # End epoch
  print("Fin epoch")
  train_loss_results.append(epoch_loss_avg.result())
  train_accuracy_results.append(epoch_accuracy.result())

  #if epoch % 50 == 0:
  print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(epoch,
                                                                epoch_loss_avg.result(),
                                                                epoch_accuracy.result()))


# In[ ]:


fig, axes = plt.subplots(2, sharex=True, figsize=(12, 8))
fig.suptitle('Training Metrics')

axes[0].set_ylabel("Loss", fontsize=14)
axes[0].plot(train_loss_results)

axes[1].set_ylabel("Accuracy", fontsize=14)
axes[1].set_xlabel("Epoch", fontsize=14)
axes[1].plot(train_accuracy_results)
plt.show()


# In[ ]:


test_url = "https://raw.githubusercontent.com/intentodemusico/Stegonomono/master/ANN/Dataset/test_70000.csv"

test_fp = tf.keras.utils.get_file(fname=os.path.basename(test_url),
                                  origin=test_url)


# In[ ]:


test_dataset = tf.data.experimental.make_csv_dataset(
    test_fp,
    batch_size,
    column_names=column_names,
    label_name='species',
    num_epochs=1,
    shuffle=False)

test_dataset = test_dataset.map(pack_features_vector)


# In[ ]:


test_accuracy = tf.keras.metrics.Accuracy()

for (x, y) in test_dataset:
  logits = model(x)
  prediction = tf.argmax(logits, axis=1, output_type=tf.int32)
  test_accuracy(prediction, y)

print("Test set accuracy: {:.3%}".format(test_accuracy.result()))


# In[ ]:


#tf.stack([y,prediction],axis=1)

