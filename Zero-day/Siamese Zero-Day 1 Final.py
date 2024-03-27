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
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, precision_score
from sklearn.utils import resample
from keras import regularizers


os.environ['PYTHONHASHSEED'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
np.random.seed(49)
rn.seed(1237)
tf.random.set_seed(65)


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


def cosine_annealing(epoch,T_max):
    
    
    eta_max = 0.024611743273392345 
    eta_min =  0.000827182726325543
    
    lr = eta_min + 0.5 * (eta_max - eta_min) * (1 + np.cos(np.pi * epoch / T_max))
    return lr


def make_pairs(X, y, similar):
    X_pairs, y_pairs = [], []
    tuples = [(x1, y1) for x1, y1 in zip(X, y)]
    classes = np.unique(y)
    print(classes)
    
    # Collect instances for each class
    class_instances = {cls: [] for cls in classes}
    
    for t in tuples:
        pack, label = t
        class_instances[label].append(pack)

    
    # Create similar pairs
    for label, instances in class_instances.items():
        length = len(instances)
        if length > 1:
            num_pairs = similar*1.5 #remeber to change if number of classes changes!!!
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
    
    # Collect instances for each class
    class_instances = {cls: [] for cls in classes}
    for t in tuples:
        pack, label = t
        class_instances[label].append(pack)
    
    
    min_length = min(len(lst) for lst in class_instances.values())
    
    
    for cls, lst in class_instances.items():
        if len(lst) > min_length:
            class_instances[cls] = lst[:min_length]

    print(class_instances)

        
    # Iterate over test instances
    for t_test in test_tuples:
        test_instance, test_label = t_test
        #polling
        for i in range(0,3):
            #Make pair with each different class
            for label, instances in class_instances.items():
                print(label,'')
                random_instance = rn.choice(instances)
                X_pairs.append([test_instance, random_instance])
                y_pairs.append(1 if label == test_label else 0)
                labels.append(test_label)

    return np.array(X_pairs), np.array(y_pairs), np.array(labels)

def convert_labels(y_pred,unseen):
    test_predict = []
    #Zero-day threshold
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

#get fixed number of samples for each class
def gettestset(X_test1, 
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


train = []
for i in range(10,11):
    df = pd.read_csv(f'C:/Users/haydc/OneDrive - University of Strathclyde/Year4/19496/Dataset (1)/Multiclass/TestDataZero/All_new{i}.csv')
    df.fillna(0, inplace=True)
    df.replace({True: 1, False: 0}, inplace=True)  
    df.drop_duplicates(ignore_index=True, inplace=True)
    train.append(df)

dataset = pd.concat(train,ignore_index=True)


y = dataset['Attack_Type']
X = dataset.drop(columns=['Attack_Type'])

X_train, X_test,  y_train, y_test = train_test_split(X, y, shuffle=True, random_state=42, test_size=0.5, stratify=y)

lab = LabelEncoder()
lab.fit(y_train)

indices_to_drop = y_train[y_train == 'SSH-Patator'].index
X_train_drop = X_train.drop(index=indices_to_drop)
y_train_drop = y_train.drop(index=indices_to_drop)

X_val = X_test
y_val = y_test


# X_train, X_test,  y_train, y_test = train_test_split(X, y, shuffle=True, random_state=42, test_size=0.5, stratify=y)

norm = StandardScaler()
norm.fit(X_train)
X_train_fin = norm.transform(X_train_drop)
X_test_fin = norm.transform(X_val)




pack_A_inp = Input(17, name='pack_A_inp')
pack_B_inp = Input(17, name='pack_B_inp')

nn = Sequential([Dense(32, activation='relu'),
                 Dense(256, activation='relu'),
                 Dense(16, activation='relu'),
                 Dense(256, activation='relu')]) 

vector_A = nn(pack_A_inp)
vector_B = nn(pack_B_inp)
#, kernel_regularizer=regularizers.l1(0.001)

output = Lambda(EuclDistance)([vector_A,vector_B])

model = Model(inputs=[pack_A_inp, pack_B_inp], outputs=output)


model.summary()

X_train_pairs, y_train_pairs = make_pairs(X_train_fin, y_train_drop, 50000)
X_test_pairs, y_test_pairs = make_pairs(X_test_fin, y_val, 5000)

num_samples = X_train_pairs.shape[0]
indices = np.arange(num_samples)
np.random.shuffle(indices)
X_train_pairs_shuffled = X_train_pairs[indices]
y_train_pairs_shuffled = y_train_pairs[indices]

print(X_train_pairs.shape)
print(X_test_pairs.shape)
# lr_schedule = ExponentialDecay(initial_learning_rate=0.0001, decay_steps=1000, decay_rate=0.9)

epoch_num = 325

lr = LearningRateScheduler(lambda epoch: cosine_annealing(epoch, epoch_num))
callbackList = [lr]

model.compile(loss=contrastive_loss,optimizer=Adam(),metrics=[calculate_accuracy])

plot_model(model, to_file='siamese.png', show_shapes=True)


history = model.fit(x=[X_train_pairs_shuffled[:,0,:],X_train_pairs_shuffled[:,1,:]],y=y_train_pairs_shuffled,epochs=epoch_num,batch_size=350000,validation_data=([X_test_pairs[:,0,:],X_test_pairs[:,1,:]],y_test_pairs),callbacks=callbackList)

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



X_test_fin = norm.transform(X_test)
y_train_fin = lab.transform(y_train_drop)
y_test_fin = lab.transform(y_test)

X_test_fin, y_test_fin = gettestset(X_test_fin,y_test_fin)
X_test_pairs, ground_truth, labels = make_pairs1(X_train_fin, y_train_fin, X_test_fin,y_test_fin)
print(X_test_pairs.shape)

predictions = model.predict(x=[X_test_pairs[:,0,:],X_test_pairs[:,1,:]])

labels1 = set(y_test_fin)
labels2 = set(y_train_fin)

# Get the unique label that is not in both sets
unseen = (labels1 ^ labels2).pop()

predict = convert_labels(predictions,unseen)
predict = polling(predict)
predict = lab.inverse_transform(predict)
y_true = lab.inverse_transform(y_test_fin.astype(np.int32))

class_metrics = classification_report(y_true, predict, output_dict=True)
cm = confusion_matrix(y_true, predict)

print(lab.inverse_transform([0,1,2,3,4]))
test = lab.inverse_transform(labels.astype(int))
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

plt.savefig(f'{unseen}_metrics_plot.png', dpi=300)

plt.show()
precision = precision_score(y_true, predict ,average='weighted')
