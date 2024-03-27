# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 10:51:22 2024

@author: haydc
"""


import matplotlib.pyplot as plt
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score,f1_score
from sklearn import tree
from sklearn.preprocessing import StandardScaler, LabelEncoder


b = 6  
train = []
for i in range(0,4):
    df = pd.read_csv(f'Filename/TestData/RandomData_{i}.csv')
    df.fillna(0, inplace=True)
    df.replace({True: 1, False: 0}, inplace=True)    
    train.append(df)

df = pd.concat(train)
X_train = df.drop(columns=['Attack_Type','Attack','TCP_Option','timestamp','ip_sourceAdd','ip_destAdd','source_port','destination_port','ip_options','flow_id','flow','Checksum','TCP_Sequence_no','ip_id','TCP_akn_no','ip_hchecksum'])
norm = StandardScaler()
norm.fit(X_train)
X_train = pd.DataFrame(norm.transform(X_train),columns=X_train.columns)

df1 = pd.read_csv(f'Filename/Multiclass/TestData/RandomData_{b}.csv')
df1.fillna(0, inplace=True)
df1.replace({True: 1, False: 0}, inplace=True)
X_test = df1.drop(columns=['Attack_Type','Attack','TCP_Option','timestamp','ip_sourceAdd','ip_destAdd','source_port','destination_port','ip_options','flow_id','flow','Checksum','TCP_Sequence_no','ip_id','TCP_akn_no','ip_hchecksum'])
X_test = pd.DataFrame(norm.transform(X_test),columns=X_test.columns)

# print(X_test.head())
lab = LabelEncoder()
y_train = lab.fit_transform(df['Attack_Type'])
y_test = lab.transform(df1['Attack_Type'])

del df
del df1

# print(y_test.head())

print("data Ready")
features = pd.read_csv('Filename/TestData/feature_selection_results_normalised_1.csv', nrows=20)
features = features['Feature']


selected_features = []
rankings =[]
for i in range(1,len(features)-1):
    print(f'{i}')
    selected_features = features.T[0:i]
    estimator = tree.DecisionTreeClassifier()
    # print(X_train[selected_features].head())
    estimator.fit(X_train[selected_features], y_train)
    
    y_pred = estimator.predict(X_test[selected_features])
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred,average='weighted')
    recall = recall_score(y_test, y_pred,average='weighted')
    f1 = f1_score(y_test,y_pred,average='weighted')
    rankings.append(pd.DataFrame({
        'no_features': i,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1': f1,
        },index=[i]))
Ranks = pd.concat(rankings, ignore_index=True)    


plt.figure()
plt.plot(Ranks['no_features'], Ranks['Accuracy'], label='Accuracy', marker='x')
plt.plot(Ranks['no_features'], Ranks['Precision'], label='Precision', marker='o')
# plt.plot(Ranks['no_features'], Ranks['Recall'], label='Recall', marker='o')
plt.plot(Ranks['no_features'], Ranks['F1'], label='F1', marker='o')
plt.xlabel('Number of Features')
plt.ylabel('Score')
plt.title('Performance Metrics vs. Number of Features')
plt.legend()

plt.show()
