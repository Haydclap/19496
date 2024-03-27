# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 11:46:52 2024

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
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.utils import plot_model
import os
import random as rn
from tensorflow.keras.callbacks import LearningRateScheduler
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, f1_score
from sklearn.utils import resample
from keras import regularizers
import optuna 
import optuna.visualization as vis

import os
import random as rn

os.environ['PYTHONHASHSEED'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
np.random.seed(49)
rn.seed(1237)

def EuclDistance(vects):
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))

def contrastive_loss(y_true, y_pred,margin=1.0):
    y_true = tf.cast(y_true, tf.float32)
    
    square_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    
    return K.mean(y_true * square_pred + (1 - y_true) * margin_square)

def calculate_accuracy(y_true, y_pred):    
    return K.mean(K.equal(y_true, K.cast(y_pred < 0.5, y_true.dtype)))


def cosine_annealing(epoch,T_max,eta_max,eta_min):
    
    # T_max = 200
    # eta_max = 0.00604  
    # eta_min =  0.000001 
    
    lr = eta_min + 0.5 * (eta_max - eta_min) * (1 + np.cos(np.pi * epoch / T_max))
    return lr


def make_pairs(X, y, similar):
    X_pairs, y_pairs = [], []
    tuples = [(x1, y1) for x1, y1 in zip(X, y)]
    classes = np.unique(y)
    # print(classes)
    
    
    class_instances = {cls: [] for cls in classes}
    
    for t in tuples:
        pack, label = t
        class_instances[label].append(pack)

    
    # Create similar pairs
    for label, instances in class_instances.items():
        length = len(instances)
        if length > 1:
            num_pairs = similar*1.5 #remeber to change if number of classes changes!!!
            # print(num_pairs,'numpairs')
            pairs_generated = 0
            for i in range(length):
                for j in range(i+1, length):
                    if pairs_generated >= num_pairs:
                        # print(pairs_generated, 'pairs generated')
                        break
                    else:
                        X_pairs.append([instances[i], instances[j]])
                        y_pairs.append(1)
                        pairs_generated += 1
                if pairs_generated >= num_pairs:
                    break
            # print(len(X_pairs),'Xpairs')
            
    # Create dissimilar pairs
    for label1, label2 in itertools.combinations(classes, 2):
        instances1 = class_instances[label1]
        instances2 = class_instances[label2]
        # print(label1,'',label2)
        length1 = len(instances1)
        length2 = len(instances2)
        min_length = min(length1, length2)
        for i in range(similar):
            idx1 = rn.randint(0, length1 - 1)
            idx2 = rn.randint(0, length2 - 1)
            X_pairs.append([instances1[idx1], instances2[idx2]])
            y_pairs.append(0)
        # print(len(X_pairs),'Xpairs')
    return np.array(X_pairs), np.array(y_pairs)

def make_pairs1(X, y, X_test, y_test):
    X_pairs, y_pairs, labels = [], [], []
    tuples = [(x, y_) for x, y_ in zip(X, y)]
    test_tuples = [(x_test, y_test_) for x_test, y_test_ in zip(X_test, y_test)]
    classes = np.unique(y)
    
    
    class_instances = {cls: [] for cls in classes}
    for t in tuples:
        pack, label = t
        class_instances[label].append(pack)
    
    
    min_length = min(len(lst) for lst in class_instances.values())
    
    
    for cls, lst in class_instances.items():
        if len(lst) > min_length:
            class_instances[cls] = lst[:min_length]

    # print(class_instances)

        
    # Iterate over test instances
    for t_test in test_tuples:
        test_instance, test_label = t_test
        
        for i in range(0,3):
            for label, instances in class_instances.items():
                # print(label,'')
                random_instance = rn.choice(instances)
                X_pairs.append([test_instance, random_instance])
                y_pairs.append(1 if label == test_label else 0)
                labels.append(test_label)

    return np.array(X_pairs), np.array(y_pairs), np.array(labels)

def convert_labels(y_pred,unseen):
    test_predict = []
    threshold = 0.04
    for i in range(0, len(y_pred), 4):
        subset = y_pred[i:i+4] 
        if (np.any(subset < threshold)):
            max_index = np.argmin(subset) 
            if(max_index >= unseen):
                max_index+=1
        else:
            max_index = unseen
        test_predict.append(max_index)
    return test_predict

def polling(prediction):
    pred = []
    for i in range(0,len(prediction),3):
        group = prediction[i:i+3]
        final_pred = max(set(group), key=group.count)
        pred.append(int(final_pred))
    return pred;

def convert_true(y_true):
    test_predict = []
    for i in range(0, len(y_true), 15):
        subset = y_true[i:i+5]  
        max_index = np.argmax(subset)  
        test_predict.append(max_index)        
    return test_predict

def gettestset(X_test1, y_test1): 
    # Combine X_test1 and y_test1 into a single array
    Xy_test1 = np.column_stack((X_test1, y_test1))
    
    
    grouped = {}
    for row in Xy_test1:
        label = row[-1]
        if label not in grouped:
            grouped[label] = []
        grouped[label].append(row)
    
    sampled_data = []
    fixed_samples_per_class = 3000 
    
    for label, group in grouped.items():
        sampled_class = resample(group, n_samples=fixed_samples_per_class, random_state=42)
        sampled_data.extend(sampled_class)
    
    sampled_array = np.array(sampled_data)
    X_test = np.delete(sampled_array, -1, axis=1)  
    y_test = sampled_array[:, -1]  
    
    return X_test, y_test

def create_siamese_network(trial):
    pack_A_inp = Input(17, name='pack_A_inp')
    pack_B_inp = Input(17, name='pack_B_inp')
    num_layers = trial.suggest_int('num_layers', 1, 5)

    layers = []

    for i in range(num_layers):
        units = trial.suggest_categorical(f'dense_{i}_units',[8, 16, 32, 64 ,128, 256])
        layers.append(Dense(units, activation='relu'))

    # layers.append(Dense(8))

    # Build the sequential model
    nn = Sequential(layers)

    vector_A = nn(pack_A_inp)
    vector_B = nn(pack_B_inp)

    output = Lambda(EuclDistance)([vector_A, vector_B])

    model = Model(inputs=[pack_A_inp, pack_B_inp], outputs=output)
    # model.summary()
    
    return model


def objective(trial):
    epoch_num = trial.suggest_categorical('no_epochs',[50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300, 325, 350, 375, 400])
    batch_size = trial.suggest_categorical('batch_size', [200000, 300000, 400000, 500000, 600000])
    eta_max = trial.suggest_float('eta_max',0.001,0.1)
    eta_min = trial.suggest_float('eta_min', 0.0000001,0.001)
    model = create_siamese_network(trial)
    
    lr = LearningRateScheduler(lambda epoch: cosine_annealing(epoch, epoch_num, eta_max, eta_min))
    callbackList = [lr]
    
    train = []
    for i in range(10,11):
        df = pd.read_csv(f'Filename/TestDataZero/All_new{i}.csv')
        df.fillna(0, inplace=True)
        df.replace({True: 1, False: 0}, inplace=True)  
        df.drop_duplicates(ignore_index=True, inplace=True)
        train.append(df)

    dataset = pd.concat(train,ignore_index=True)


    y = dataset['Attack_Type']
    #X = dataset.drop(columns=['Attack_Type','Attack'])
    X = dataset.drop(columns=['Attack_Type'])
    #,'AtoB_avg_time','bytes_ratio','Total_TCP_flag_ACK','Total_TCP_flag_SYN','BtoA_min_inter_time','AtoB_max_len','BtoA_mean_pkt_len','BtoA_std_pkt_len','AtoB_mean_pkt_len'
    X_train, X_test,  y_train, y_test = train_test_split(X, y, shuffle=True, random_state=42, test_size=0.5, stratify=y)

    lab = LabelEncoder()
    lab.fit(y_train)

    indices_to_drop = y_train[y_train == Attack_Type].index
    X_train_drop = X_train.drop(index=indices_to_drop)
    y_train_drop = y_train.drop(index=indices_to_drop)

    X_val = X_test
    y_val = y_test


    # X_train, X_test,  y_train, y_test = train_test_split(X, y, shuffle=True, random_state=42, test_size=0.5, stratify=y)

    norm = StandardScaler()
    norm.fit(X_train)
    X_train_fin = norm.transform(X_train_drop)
    X_test_fin = norm.transform(X_val)

    
    X_train_pairs, y_train_pairs = make_pairs(X_train_fin, y_train_drop, 50000)
    X_test_pairs, y_test_pairs = make_pairs(X_test_fin, y_val, 5000)

    num_samples = X_train_pairs.shape[0]
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    X_train_pairs_shuffled = X_train_pairs[indices]
    y_train_pairs_shuffled = y_train_pairs[indices]

    print(X_train_pairs.shape)
    print(X_test_pairs.shape)
    
    model.compile(loss=contrastive_loss,optimizer=Adam(),metrics=[calculate_accuracy])
    
    history = model.fit(x=[X_train_pairs_shuffled[:,0,:],X_train_pairs_shuffled[:,1,:]],
                        y=y_train_pairs_shuffled,
                        epochs=epoch_num,
                        batch_size=batch_size,
                        validation_data=([X_test_pairs[:,0,:],X_test_pairs[:,1,:]],y_test_pairs),
                        callbacks=callbackList,
                        verbose=0)
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



    X_test_fin = norm.transform(X_test)
    y_train_fin = lab.transform(y_train_drop)
    y_test_fin = lab.transform(y_test)

    X_test_fin, y_test_fin = gettestset(X_test_fin,y_test_fin)
    X_test_pairs, ground_truth, labels = make_pairs1(X_train_fin, y_train_fin, X_test_fin,y_test_fin)
    print(X_test_pairs.shape)

    predictions = model.predict(x=[X_test_pairs[:,0,:],X_test_pairs[:,1,:]])

    labels1 = set(y_test_fin)
    labels2 = set(y_train_fin)

    unseen = (labels1 ^ labels2).pop()

    predict = convert_labels(predictions,unseen)
    predict = polling(predict)
    predict = lab.inverse_transform(predict)
    y_true = lab.inverse_transform(y_test_fin.astype(np.int32))

    class_metrics = classification_report(y_true, predict, output_dict=True)
    cm = confusion_matrix(y_true, predict)

    print(lab.inverse_transform([0,1,2,3,4]))
    # test = lab.inverse_transform(labels.astype(int))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.show()
    precision = precision_score(y_true, predict ,average='weighted')
    recall = recall_score(y_true, predict, average='weighted')
    f1 = f1_score(y_true, predict, average='weighted')
    accuracy = class_metrics['accuracy']
    print(f'Precision: {precision}, Recall: {recall}, F1: {f1}, Accuracy: {accuracy}')
    return 0.4*accuracy+0.4*f1+0.2*recall

trials = []
#'DoS Hulk', 'DoS slowloris', 'FTP-Patator', 'SSH-Patator'
attacks = ['DoS Hulk', 'DoS slowloris', 'FTP-Patator', 'SSH-Patator']
for attack in attacks:
    Attack_Type = attack
    study = optuna.create_study(direction='maximize', storage=f'sqlite:///{attack}_1.db')
    study.optimize(objective, n_trials=25)
    trials.append(study)

for a in trials:
    optuna.visualization.plot_optimization_history(a)
    optuna.visualization.plot_slice(a)
    fig = optuna.visualization.plot_param_importances(a)
    fig.show()
    best_trial = a.best_trial
    best_params = best_trial.params

    print("Best Parameters:")
    for param_name, param_value in best_params.items():
        print(f"{param_name}: {param_value}")     
