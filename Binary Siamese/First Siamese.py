# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 12:17:50 2024

@author: haydc
"""


import itertools
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential, Model
from sklearn.preprocessing import StandardScaler,LabelEncoder
from keras.optimizers import SGD, Adam

from tensorflow.keras.utils import plot_model




def make_pairs (X,y):
    X_pairs, y_pairs = [],[]
    
    tuples = [(x1,y1) for x1, y1 in zip(X, y)]
    
    for t in itertools.product(tuples, tuples):
        pair_A, pair_B = t
        pack_A, label_A = t[0]
        pack_B, label_B = t[1]

        new_label = int(label_A == label_B)
        
        X_pairs.append([pack_A, pack_B])
        y_pairs.append(new_label)
  
    X_pairs = np.array(X_pairs)
    y_pairs = np.array(y_pairs)

    return X_pairs, y_pairs


train = []
for i in range(0,1):
    df = pd.read_csv(f'Fileneame/TestData/RandomData_{i}_elim1.csv')
    df.fillna(0, inplace=True)
    df.replace({True: 1, False: 0}, inplace=True)    
    train.append(df)

dataset = pd.concat(train,ignore_index=True)


y = dataset['Attack_Type']
X = dataset.drop(columns=['Attack_Type','Attack'])


X_train, X_test,  y_train, y_test = train_test_split(X, y, shuffle=True, random_state=42, test_size=0.95, stratify=y)
norm = StandardScaler()
norm.fit(X_train)
X_train_fin = norm.transform(X_train)
X_test_fin = norm.transform(X_test)



pack_A_inp = Input(11, name='pack_A_inp')
pack_B_inp = Input(11, name='pack_B_inp')

nn = Sequential([Dense(9, activation='relu'),
                   Dense(6, activation='relu'),
                   Dense(3, activation='relu'),
                   Dense(15, activation='relu')]
                   ) 

vector_A = nn(pack_A_inp)
vector_B = nn(pack_B_inp)

concat = Concatenate()([vector_A,vector_B])

dense = Dense(15, activation='relu')(concat)
output = Dense(1, activation='sigmoid')(dense)

model = Model(inputs=[pack_A_inp, pack_B_inp], outputs=output)


model.summary()

X_train_pairs, y_train_pairs = make_pairs(X_train_fin, y_train)
print(X_train_pairs.shape)
print(X_train_pairs.shape)

model.compile(loss='binary_crossentropy',optimizer=Adam(learning_rate=0.001),metrics=['accuracy'])

plot_model(model, to_file='siamese.png', show_shapes=True)
history = model.fit(x=[X_train_pairs[:,0,:],X_train_pairs[:,1,:]],y=y_train_pairs,epochs=10,batch_size=128)


