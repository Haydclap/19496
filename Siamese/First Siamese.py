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
from tensorflow.keras.callbacks import LearningRateScheduler
from sklearn.metrics import classification_report, confusion_matrix,  ConfusionMatrixDisplay, precision_score


def EuclDistance(vects):
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))

def contrastive_loss(y_true, y_pred,margin=0.7):
    y_true = tf.cast(y_true, tf.float32)
    
    square_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    
    return K.mean(y_true * square_pred + (1 - y_true) * margin_square)

def calculate_accuracy(y_true, y_pred):    
    return K.mean(K.equal(y_true, K.cast(y_pred < 0.5, y_true.dtype)))


def cosine_annealing(epoch,T_max):
    
    eta_max = 0.00604  
    eta_min =  0.000001 
    
    lr = eta_min + 0.5 * (eta_max - eta_min) * (1 + np.cos(np.pi * epoch / T_max))
    return lr


def make_pairs(X, y, similar):
    X_pairs, y_pairs = [], []
    tuples = [(x1, y1) for x1, y1 in zip(X, y)]
    classes = np.unique(y)
    print(classes)
    
    
    class_instances = {cls: [] for cls in classes}
    
    for t in tuples:
        pack, label = t
        class_instances[label].append(pack)

    
    # Create similar pairs
    for label, instances in class_instances.items():
        length = len(instances)
        if length > 1:
            num_pairs = similar*2
            print(num_pairs,'numpairs')
            pairs_generated = 0
            for i in range(length):
                for j in range(i+1, length):
                    if pairs_generated >= num_pairs:
                        print(pairs_generated, 'pairs generated')
                        break
                    else:
                        X_pairs.append([instances[i], instances[j]])
                        y_pairs.append(1)
                        pairs_generated += 1
                if pairs_generated >= num_pairs:
                    break
            print(len(X_pairs),'Xpairs')
            
    # Create dissimilar pairs
    for label1, label2 in itertools.combinations(classes, 2):
        instances1 = class_instances[label1]
        instances2 = class_instances[label2]
        print(label1,'',label2)
        length1 = len(instances1)
        length2 = len(instances2)
        min_length = min(length1, length2)
        for i in range(similar):
            idx1 = rn.randint(0, length1 - 1)
            idx2 = rn.randint(0, length2 - 1)
            X_pairs.append([instances1[idx1], instances2[idx2]])
            y_pairs.append(0)
        print(len(X_pairs),'Xpairs')
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

    # Iterate over test instances
    for t_test in test_tuples:
        test_instance, test_label = t_test
        
        for i in range(0,3):
            for label, instances in class_instances.items():
                random_instance = rn.choice(instances)
                X_pairs.append([test_instance, random_instance])
                y_pairs.append(1 if label == test_label else 0)
                labels.append(test_label)

    return np.array(X_pairs), np.array(y_pairs), np.array(labels)

def convert_labels(y_pred):
    test_predict = []
    for i in range(0, len(y_pred), 5):
        subset = y_pred[i:i+5]  
        max_index = np.argmin(subset)
        test_predict.append(max_index)
    return test_predict

def polling(prediction):
    pred = []
    for i in range(0,len(prediction),3):
        group = prediction[i:i+3]
        final_pred = max(set(group), key=group.count)
        pred.append(final_pred)
    return pred;

def convert_true(y_true):
    test_predict = []
    for i in range(0, len(y_true), 15):
        subset = y_true[i:i+5]  
        max_index = np.argmax(subset)  
        test_predict.append(max_index)        
    return test_predict

train = []
for i in range(0,1):
    df = pd.read_csv(f'FILENAME/TestData/Final{i}.csv')
    df.fillna(0, inplace=True)
    df.replace({True: 1, False: 0}, inplace=True)  
    df.drop_duplicates(ignore_index=True, inplace=True)
    train.append(df)

dataset = pd.concat(train,ignore_index=True)


y = dataset['Attack_Type']
X = dataset.drop(columns=['Attack_Type','Attack'])


X_train, X_test,  y_train, y_test = train_test_split(X, y, shuffle=True, random_state=42, test_size=0.5, stratify=y)

norm = StandardScaler()
norm.fit(X_train)
X_train_fin = norm.transform(X_train)
X_test_fin = norm.transform(X_test)




pack_A_inp = Input(26, name='pack_A_inp')
pack_B_inp = Input(26, name='pack_B_inp')

nn = Sequential([Dense(64, activation='relu'),
                 Dropout(0.1),
                 Dense(256, activation='relu'),
                 Dropout(0.1),
                 Dense(16, activation='relu'),
                 Dense(8)]) 

vector_A = nn(pack_A_inp)
vector_B = nn(pack_B_inp)


output = Lambda(EuclDistance)([vector_A,vector_B])

model = Model(inputs=[pack_A_inp, pack_B_inp], outputs=output)


model.summary()

X_train_pairs, y_train_pairs = make_pairs(X_train_fin, y_train, 1500)
X_test_pairs, y_test_pairs = make_pairs(X_test_fin, y_test, 1500)

print(X_train_pairs.shape)
print(X_test_pairs.shape)
# lr_schedule = ExponentialDecay(initial_learning_rate=0.0001, decay_steps=1000, decay_rate=0.9)


epoch_num = 300

lr = LearningRateScheduler(lambda epoch: cosine_annealing(epoch, epoch_num))
callbackList = [lr]

model.compile(loss=contrastive_loss,optimizer=Adam(),metrics=[calculate_accuracy])

plot_model(model, to_file='siamese.png', show_shapes=True)


history = model.fit(x=[X_train_pairs[:,0,:],X_train_pairs[:,1,:]],y=y_train_pairs,epochs=epoch_num,batch_size=30000,validation_data=([X_test_pairs[:,0,:],X_test_pairs[:,1,:]],y_test_pairs),callbacks=callbackList)

model.save_weights('Multiclass_Siamese.h5')

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

lab = LabelEncoder()
y_train_fin = lab.fit_transform(y_train)
y_test_fin = lab.transform(y_test)

# index_benign = np.where(y_test_fin == 0)[0]
# index_benign_sampled = np.random.choice(index_benign, size=80, replace=False)

# X_benign = X_test_fin[index_benign_sampled]
# index_attack = np.where(y_test_fin != 0)[0]
# X_attack = X_test_fin[index_attack]

# y_test_fin = np.concatenate([y_test_fin[index_benign_sampled], y_test_fin[index_attack]])
# X_test_fin = np.concatenate([X_benign, X_attack])


X_test_pairs, ground_truth, labels = make_pairs1(X_train_fin, y_train_fin, X_test_fin,y_test_fin)
print(X_test_pairs.shape)

predictions = model.predict(x=[X_test_pairs[:,0,:],X_test_pairs[:,1,:]])

predict = convert_labels(predictions)
predict = polling(predict)
predict = lab.inverse_transform(predict)
y_true = lab.inverse_transform(y_test_fin)

class_metrics = classification_report(y_true, predict, output_dict=True)
cm = confusion_matrix(y_true, predict)

print(lab.inverse_transform([0,1,2,3,4]))
test = lab.inverse_transform(labels)

disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()

classes = list(class_metrics.keys())[:-3]  


precision = []
recall = []
f1_score = []


for cls in classes:
    precision.append(class_metrics[cls]['precision'])
    recall.append(class_metrics[cls]['recall'])
    f1_score.append(class_metrics[cls]['f1-score'])


x = np.arange(len(classes))  
width = 0.25

fig, ax = plt.subplots()
bars_precision = ax.bar(x - width, precision, width, label='Precision')
bars_recall = ax.bar(x, recall, width, label='Recall')
bars_f1_score = ax.bar(x + width, f1_score, width, label='F1-score')


ax.set_xlabel('Classes')
ax.set_ylabel('Scores')
ax.set_title('Metrics by Class')
ax.set_xticks(x)
ax.set_xticklabels(classes, rotation=45)
ax.legend(loc='upper left', bbox_to_anchor=(1, 1))

fig.tight_layout()

plt.savefig(f'Siamese_Classification_metrics_plot.png', dpi=300)

plt.show()
precision = precision_score(y_true, predict ,average='weighted')




