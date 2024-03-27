# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 16:11:04 2024

@author: haydc
"""

import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Reshape
from tensorflow.keras.metrics import Accuracy, Precision, Recall
import matplotlib.pyplot as plt
from keras.optimizers import SGD, Adam
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.utils import shuffle
from sklearn.model_selection import StratifiedKFold
import os
import random as rn
os.environ['PYTHONHASHSEED'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
np.random.seed(37)
rn.seed(1254)
tf.random.set_seed(89)




i=1
dataset = np.loadtxt(f'FILENAME/TestData/RandomData_{i}_elim2.csv', delimiter=',',skiprows=1)
X = dataset[:,0:9]
y = dataset[:,9]


stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold_losses = []
fold_valLosses = []
accuracy_scores = []
precision_scores = []
recall_scores = []



for  fold_id, (train_index, test_index) in enumerate(stratified_kfold.split(X, y)):
    X_train = X[train_index]
    X_test = X[test_index]
    y_train = y[train_index]
    y_test = y[test_index]

    norm = StandardScaler()
    norm.fit(X_train)
    X_train_fin = norm.transform(X_train)
    X_test_fin = norm.transform(X_test)
    
    model = Sequential() 
    model.add(Dense(18,input_shape=(9,), activation='relu'))
    model.add(Dense(12, activation='relu'))
    model.add(Dense(9, activation='relu'))
    model.add(Dense(6, activation='relu'))
    model.add(Dense(3, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    optimizer = Adam(learning_rate=0.001)
    
    
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy', Precision(), Recall()],)
    
    # X_train_fin1, y_train = shuffle(X_train_fin, y_train, random_state=42)
    # X_test_fin1 , y_test = shuffle(X_test_fin, y_test, random_state=42)
    
    history = model.fit(X_train_fin, y_train, epochs=400, batch_size=32,validation_data=(X_test_fin, y_test))
    
    fold_losses.append(history.history['loss'])
    fold_valLosses.append(history.history['val_loss'])
    
    plt.figure()
    plt.plot(history.history['loss'], label=f'Fold {fold_id + 1} Train')
    plt.plot(history.history['val_loss'], label=f'Fold {fold_id + 1} Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Fold {fold_id + 1} Training and Validation Losses')
    plt.legend()
    plt.show()
    
    i=0
    dataset = np.loadtxt(f'C:/Users/haydc/OneDrive - University of Strathclyde/Year4/19496/Dataset (1)/CSV_data/TestData/RandomData_{i}_elim2.csv', delimiter=',',skiprows=1)
    X_val = dataset[:,0:9]
    y_val = dataset[:,9]
    
    X_val = norm.transform(X_val)
    _, accuracy, precision, recall = model.evaluate(X_val, y_val)
    accuracy_scores.append(accuracy)
    precision_scores.append(precision)
    recall_scores.append(recall)
    
print('precision :', precision_scores)
print('accuracy :', accuracy_scores)
print('recall: ', recall_scores)
    
