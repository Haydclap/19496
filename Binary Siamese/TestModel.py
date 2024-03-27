# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 11:29:17 2024

@author: haydc
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
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report

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


def make_pairs(X, y, Attacks, Benign, X_test, y_test):
    X_pairs, ground_truth = [], []

    # Extract indices of attack and benign samples from y
    attack_indices = np.where(y == 1)[0]
    benign_indices = np.where(y == 0)[0]

    # Shuffle indices to ensure randomness
    np.random.shuffle(attack_indices)
    np.random.shuffle(benign_indices)

    for i in range(600):
        attack_index = attack_indices[i % len(attack_indices)]
        benign_index = benign_indices[i % len(benign_indices)]
    
        # Create attack pair
        attack_pair = [X[attack_index], X_test[i]]
        X_pairs.append(attack_pair)
        ground_truth.append([0, 1])  # One-hot encoded ground truth: [0, 1] for attack
    
        # Create benign pair
        benign_pair = [X[benign_index], X_test[i]]
        X_pairs.append(benign_pair)
        ground_truth.append([1, 0])  # One-hot encoded ground truth: [1, 0] for benign

    X_pairs = np.array(X_pairs)
    ground_truth = np.array(ground_truth)

    return X_pairs, ground_truth



def get_model():
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
    return model


train = []
for i in range(1,2):
    df = pd.read_csv(f'FILENAME/TestData/RandomData_{i}_elim2.csv')
    df.drop_duplicates()
    df.fillna(0, inplace=True)
    df.replace({True: 1, False: 0}, inplace=True)    
    train.append(df)

dataset = pd.concat(train,ignore_index=True)


y = dataset['Attack']
X = dataset.drop(columns=['Attack'])

X_train, X_test1,  y_train, y_test1 = train_test_split(X, y, shuffle=True, random_state=42, test_size=0.6, stratify=y)
X_test, X_rest, y_test, y_rest = train_test_split(X_test1,y_test1,shuffle=True, random_state=42, test_size=0.6, stratify=y_test1)
norm = StandardScaler()
norm.fit(X_train)
X_train_fin = norm.transform(X_train)
X_test_fin = norm.transform(X_test)



X_train_pairs, ground_truth = make_pairs(X_train_fin, y_train, 400, 400,X_test_fin,y_test)

model = get_model()
model.load_weights('Binary_Siamese.h5')

predictions = model.predict(x=[X_train_pairs[:,0,:],X_train_pairs[:,1,:]])
y_pred = (predictions > 0.5).astype(int)
y_true = np.argmax(ground_truth, axis=1)
class_metrics = classification_report(y_true, y_pred, output_dict=True)

