# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 10:09:13 2024

@author: haydc
"""


from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score,f1_score
from sklearn import tree
from sklearn.preprocessing import StandardScaler, LabelEncoder


def CalcMetrics(i,current_attack):
    
    df = pd.read_csv(f'Filename/TestDataZero/RFE1.csv')
    #df.fillna(df.mean(), inplace=True)    
    df.replace({True: 1, False: 0}, inplace=True)
    
    
    print("data Ready")
    #'TCP_flag_FIN','TCP_flag_SYN','TCP_flag_RST','TCP_flag_PUSH','TCP_flag_ACK','TCP_flag_URG','TCP_flag_ECE','TCP_flag_CWR'
    
    
    
    
    # X = df.drop(columns=['Attack','TCP_Option','timestamp','ip_sourceAdd','ip_destAdd','source_port','destination_port','ip_options','TCP_flag_FIN','TCP_flag_SYN','TCP_flag_RST','TCP_flag_PUSH','TCP_flag_ACK','TCP_flag_URG','TCP_flag_ECE','TCP_flag_CWR','flow_id','flow'])
    y_train = df['Attack_Type']
    attacks = ['Benign','DoS Hulk','DoS slowloris','FTP-Patator','SSH-Patator']
    attacks.remove(current_attack)
    
    
    indices_to_drop = y_train[y_train == attacks[0]].index
    X_train_drop = df.drop(index=indices_to_drop)
    y_train_drop = y_train.drop(index=indices_to_drop)
    
    
    indices_to_drop = y_train[y_train == attacks[1]].index
    X_train_drop = df.drop(index=indices_to_drop)
    y_train_drop = y_train.drop(index=indices_to_drop)
    
    indices_to_drop = y_train[y_train == attacks[2]].index
    X_train_drop = df.drop(index=indices_to_drop)
    y_train_drop = y_train.drop(index=indices_to_drop)
    
    indices_to_drop = y_train[y_train == attacks[3]].index
    X_train_drop = df.drop(index=indices_to_drop)
    y_train_drop = y_train.drop(index=indices_to_drop)
    
    
    data = X_train_drop.drop(columns=['Attack_Type','Attack','TCP_Option','timestamp','ip_sourceAdd','ip_destAdd','source_port','destination_port','ip_options','flow_id','flow','Checksum','TCP_Sequence_no','ip_id','TCP_akn_no','ip_hchecksum'])
    data.fillna(0, inplace=True)
    
    norm = StandardScaler()
    norm.fit(data)
    X = pd.DataFrame(norm.transform(data),columns=data.columns)
    
    lab = LabelEncoder()
    y = lab.fit_transform(y_train_drop)
    
    
    
 
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42,stratify=y)
    print('here')
    estimator = tree.DecisionTreeClassifier()  
    rfe = RFE(estimator, n_features_to_select=1)
    rfe.fit(X, y)
    selected_features = X.columns[rfe.support_]
    print("Selected Features:", selected_features)
    estimator.fit(X[selected_features], y)
        
    y_train_pred = estimator.predict(X[selected_features])
    train_accuracy = accuracy_score(y, y_train_pred)
    print(f"Training Accuracy: {train_accuracy}")
    
    feature_rankings = rfe.ranking_
    
    rankings_df = pd.DataFrame({
        'Feature': X.columns,
        'Ranking': feature_rankings
    })
    rankings_df = rankings_df.sort_values('Feature')
    if(i>0):
        rankings_df.drop('Feature',axis=1)
    
    return rankings_df
    
    


#features = []  
attacks = ['Benign','DoS Hulk','DoS slowloris','FTP-Patator','SSH-Patator']
df = pd.DataFrame(CalcMetrics(0,attacks[0]))
for i in range(1,4):
    print(i)
    Ranking = CalcMetrics(i,attacks[i])
    df[attacks[i]] = Ranking['Ranking']
    #features.append(Ranking)
    print(i," complete")

# rankings_df = pd.concat(features,ignore_index=True)
df['Feature Averages'] = df.mean(numeric_only=True, axis=1)
df = df.sort_values('Feature Averages')
df.to_csv('Filename/TestDataZero/feature_all_individual.csv', index=False)


#nan_values = data.isna().sum()
