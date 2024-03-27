# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 11:08:31 2024

@author: haydc

Siamese network First Attempt
"""

import itertools
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold, train_test_split
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential, Model
from sklearn.preprocessing import StandardScaler,LabelEncoder
from keras.optimizers import SGD, Adam
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.utils import plot_model
import os
import random as rn
os.environ['PYTHONHASHSEED'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
np.random.seed(49)
rn.seed(1237)
tf.random.set_seed(65)


def EuclDistance(vects):
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))
    

def contrastive_loss(y_true, y_pred,margin=1):
    y_true = tf.cast(y_true, tf.float32)
    
    square_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    
    return K.mean(y_true * square_pred + (1 - y_true) * margin_square)

def calculate_accuracy(y_true, y_pred):    
    return K.mean(K.equal(y_true, K.cast(y_pred < 0.5, y_true.dtype)))


def make_pairs (X,y):
    X_pairs, y_pairs = [],[]
    benignpairs, attackpairs, ylabB, ylabA = [],[],[],[]
    tuples = [(x1,y1) for x1, y1 in zip(X, y)]
    acount=0
    bcount=0
    bothcount=0
    Attacks = 400
    Benign = 400
    A, B = [],[]
    A_y, B_y = [], []
    
    
    for t in itertools.product(tuples, tuples):
        pair_A, pair_B = t
        pack_A, label_A = t[0]
        pack_B, label_B = t[1]

        new_label = int(label_A == label_B)
        if new_label == 1:
            if label_A == 0:
                B.append([pack_A, pack_B])
                B_y.append(new_label)
            else:
                A.append([pack_A, pack_B])
                A_y.append(new_label)
                acount += 1
        else:
            X_pairs.append([pack_A, pack_B])
            y_pairs.append(new_label)
            
    
    pick = rn.sample(range(len(X_pairs)), Attacks*2)
    X_pairs = [X_pairs[i] for i in pick] + A[:Attacks] + B[:Benign]
    y_pairs = [y_pairs[i] for i in pick] + [1]*Attacks + [1]*Benign
    
    return np.array(X_pairs), np.array(y_pairs)
    


train = []
for i in range(0,1):
    df = pd.read_csv(f'FILENAME/TestData/RandomData_{i}_elim2.csv')
    df.drop_duplicates()
    df.fillna(0, inplace=True)
    df.replace({True: 1, False: 0}, inplace=True)    
    train.append(df)

dataset = pd.concat(train,ignore_index=True)


y = dataset['Attack']
X = dataset.drop(columns=['Attack'])


X_train, X_test1,  y_train, y_test1 = train_test_split(X, y, shuffle=True, random_state=42, test_size=0.95, stratify=y)
X_test, X_rest, y_test, y_rest = train_test_split(X_test1,y_test1,shuffle=True, random_state=42, test_size=0.95, stratify=y_test1)
norm = StandardScaler()
norm.fit(X_train)
X_train_fin = norm.transform(X_train)
X_test_fin = norm.transform(X_test)



pack_A_inp = Input(9, name='pack_A_inp')
pack_B_inp = Input(9, name='pack_B_inp')

nn = Sequential([Dense(128, activation='relu'),
                   Dense(128, activation='relu'),
                   Dense(128, activation='relu')]
                   )

vector_A = nn(pack_A_inp)
vector_B = nn(pack_B_inp)



output = Lambda(EuclDistance)([vector_A,vector_B])
# normal_layer = BatchNormalization()(output)
# output_layer = Dense(1, activation="sigmoid")(normal_layer)
model = Model(inputs=[pack_A_inp, pack_B_inp], outputs=output)


model.summary()

X_train_pairs, y_train_pairs = make_pairs(X_train_fin, y_train)
X_test_pairs, y_test_pairs = make_pairs(X_test_fin, y_test)

# lr_schedule = ExponentialDecay(initial_learning_rate=0.01, decay_steps=10000, decay_rate=0.9)

model.compile(loss=contrastive_loss,optimizer=Adam(learning_rate=0.0001),metrics=[calculate_accuracy])

plot_model(model, to_file='siamese.png', show_shapes=True)
history = model.fit(x=[X_train_pairs[:,0,:],X_train_pairs[:,1,:]],y=y_train_pairs,epochs=800,batch_size=16,validation_data=([X_test_pairs[:,0,:],X_test_pairs[:,1,:]],y_test_pairs))

model.save_weights('Binary_Siamese.h5')

plt.figure()
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Validation')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Losses')
plt.legend()
plt.grid(True)
plt.show()

plt.figure()
plt.plot(history.history['calculate_accuracy'], label='Train')
plt.plot(history.history['val_calculate_accuracy'], label='Validation')
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.grid(True)
plt.show()




