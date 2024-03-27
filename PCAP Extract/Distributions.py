# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 11:09:09 2024

@author: haydc
"""

import pandas as pd


days = ['Wednesday','Thursday','Friday']
for i in range(2,4):
    df = pd.read_csv(f'C:/Users/haydc/OneDrive - University of Strathclyde/Year4/19496/Dataset (1)/Multiclass/TestData/RandomData_{i}_equal.csv', encoding='utf-8', encoding_errors='replace')
    total_distribution = df['Attack_Type'].value_counts()
    print(total_distribution)
    # print(df[(df['ip_destAdd']=='192.168.10.50') & (df['ip_sourceAdd']=='172.16.0.11')])
    