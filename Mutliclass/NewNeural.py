# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 12:24:33 2023

@author: haydc
"""

import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Reshape
from tensorflow.keras.metrics import Accuracy, Precision, Recall
import matplotlib.pyplot as plt
from keras.optimizers import SGD, Adam
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.utils import shuffle
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.utils import plot_model



train = []
for i in range(0,1):
    df = pd.read_csv(f'Filename/TestData/Final{i}.csv')
    df.fillna(0, inplace=True)
    df.replace({True: 1, False: 0}, inplace=True)
    df.drop_duplicates(ignore_index=True, inplace=True)
    train.append(df)

dataset = pd.concat(train,ignore_index=True)
dataset.drop_duplicates(ignore_index=True, inplace=True)

y = dataset['Attack_Type']
X = dataset.drop(columns=['Attack_Type','Attack','AtoB_max_len','AtoB_mean_offset','BtoA_avg_time','AtoB_mean_pkt_len','AtoB_time_std','BtoA_time_std','BtoA_max_inter_time','BtoA_std_pkt_len','bytes_AtoB','BtoA_min_inter_time','packets_AtoB','bytes_BtoA','BtoA_max_len','BtoA_mean_pkt_len','packets_BtoA','AtoB_flag_PUSH','AtoB_min_inter_time','AtoB_min_len','BtoA_min_len'])


stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold_losses = []
fold_valLosses = []
accuracy_scores = []
precision_scores = []
recall_scores = []
confusion =[]
reports = []

for  fold_id, (train_index, test_index) in enumerate(stratified_kfold.split(X, y)):
    X_train = X.iloc[train_index, :]
    X_test = X.iloc[test_index, :]
    y_train = y.iloc[train_index]
    y_test = y.iloc[test_index]

    norm = StandardScaler()
    norm.fit(X_train)
    X_train_fin = norm.transform(X_train)
    X_test_fin = norm.transform(X_test)
    lab = LabelEncoder()
    y_train = lab.fit_transform(y_train)
    y_test = lab.transform(y_test)
    y_train = to_categorical(y_train, num_classes=5)
    y_test = to_categorical(y_test, num_classes=5)
    model = Sequential() 
    model.add(Dense(6,input_shape=(7,), activation='relu'))
    # model.add(Dense(6, activation='relu'))
    # model.add(Dense(3, activation='relu'))
    model.add(Dense(5, activation='softmax'))
    
    optimizer = Adam(learning_rate=0.001)
    
    
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy', Precision(), Recall()],)
    plot_model(model, to_file='Neural.png', show_shapes=True)
    # X_train_fin1, y_train = shuffle(X_train_fin, y_train, random_state=42)
    # X_test_fin1 , y_test = shuffle(X_test_fin, y_test, random_state=42)
    
    history = model.fit(X_train_fin, y_train, epochs=400, batch_size=32,validation_data=(X_test_fin, y_test))
    
    y_pred = np.around(model.predict(X_test_fin))
    # y_pred = lab.inverse_transform(y_pred)
    # y_test = lab.inverse_transform(np.argmax(y_test,axis=1))
    report = classification_report(y_test, y_pred)
    
    
    print(report)
    
    
    class_metrics = classification_report(y_test, y_pred, output_dict=True)
        
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
    plt.figure()
    precision = []
    re = []
    f1=[]
    sup=[]
    
    attacks = ['Benign','DoS Hulk','DoS slowloris','FTP-Patator','SSH-Patator']
    for a in range(0,5):
        precision.append(class_metrics[f'{a}']['precision'])
        re.append(class_metrics[f'{a}']['recall'])
        f1.append(class_metrics[f'{a}']['f1-score'])
        sup.append(class_metrics[f'{a}']['support'])
    bar_width = 0.25
    classes = np.arange(len(precision))
    
    
    y_test1 = lab.inverse_transform(np.argmax(y_test, axis=1))
    y_pred1 = lab.inverse_transform(np.argmax(y_pred, axis=1))
    confusion.append(confusion_matrix(y_test1, y_pred1))
    class_metrics = classification_report(y_test1, y_pred1, output_dict=True)
    reports.append(class_metrics)

    
    fig, ax = plt.subplots()
    bar1 = ax.bar(classes - bar_width, precision, width=bar_width, label='Precision')
    bar2 = ax.bar(classes, re, width=bar_width, label='Recall')
    bar3 = ax.bar(classes + bar_width, f1, width=bar_width, label='F1-score')
    ax.set_xticks(classes)
    ax.set_xticklabels(lab.inverse_transform(classes),rotation=45, ha='right') 
    plt.legend()
    plt.show()
    
