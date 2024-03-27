# -*- coding: utf-8 -*-
"""
Created on Sun Jan 28 21:50:14 2024

@author: haydc
"""
import pandas as pd
import glob

days = ['Tuesday','Wednesday']
for day in days:
    files = glob.glob(f'FileName/Flows/{day}/*.csv')
    fileLen = len(files)
    names = []
    for i in range(0,fileLen):
        print(f'{day}_{i}')
        df = pd.read_csv(f'FileName/Multiclass/Flows/{day}/{day}_flows_{i}.csv')
        
        
        df_filtered = df[df['Attack_Type'] != 'Benign']
        del df
        
        grouped = df_filtered.groupby('Attack_Type')
        for attack_type, group_df in grouped:
            print(f'{attack_type}_{i}')
            if attack_type not in names:
                
                print(f'{attack_type}')
                output_file = f'C:/Users/haydc/OneDrive - University of Strathclyde/Year4/19496/Dataset (1)/Multiclass/Attacks/{attack_type}_flows.csv'
                group_df.to_csv(output_file, index=False, encoding='utf-8')
            else:
                print(f'{attack_type}')
                output_file = f'C:/Users/haydc/OneDrive - University of Strathclyde/Year4/19496/Dataset (1)/Multiclass/Attacks/{attack_type}_flows.csv'
                group_df.to_csv(output_file, index=False, encoding='utf-8', mode='a', header=False)
            names.append(attack_type)
    
