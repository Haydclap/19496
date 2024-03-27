# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 14:44:02 2023

@author: haydc
"""

import pandas as pd
import glob
import random
from math import floor

dtypes = {
    'timestamp': str,
    'ip_v' : int,
    'ip_hl': int,
    'ip_tos': int,
    'ip_tl': int,
    'ip_id': int,
    'ip_flag_rf': int,
    'ip_flag_df': int,
    'ip_flag_mf': int,
    'ip_fragoff': int,
    'ip_ttl': int,
    'ip_protocol': int,
    'ip_hchecksum': int,
    'ip_sourceAdd': str,
    'ip_destAdd': str,
    'ip_options':str,
    'source_port': int,
    'destination_port': int,
    'TCP_Sequence_no': float,
    'TCP_akn_no': float,
    'TCP_DO': int,
    'TCP_flags': int,
    'TCP_flag_FIN': bool,
    'TCP_flag_SYN': bool,
    'TCP_flag_RST': bool,
    'TCP_flag_PUSH': bool,
    'TCP_flag_ACK': bool,
    'TCP_flag_URG': bool,
    'TCP_flag_ECE': bool,
    'TCP_flag_CWR': bool,
    'TCP_window': int,
    'Checksum': int,
    'TCP_urgent_pointer': int,
    'TCP_Option': str,
    'UDP_length': int,      
    'Attack': bool,
}


def select_data(day):
    files = glob.glob(f'FILENAME/CSV_data/{day}/Flows/*.csv')
    fileLen = len(files)
    count = 0
    print(fileLen)
    print(day)
    file_no = random.randint(0,fileLen-1)
    data_no = random.randint(1,400000-100)
    df = pd.read_csv(f'FILENAME/CSV_data/{day}/Flows/{day}_flows_{file_no}.csv',skiprows=range(1,data_no),nrows=100,dtype=dtypes)
    df_attack = df[df['Attack']== True]
    df_benign = df[df['Attack'] == False]
    count += 1
    print(count)
    return df_attack,df_benign
   

def TestSelector(i):     
    days = ['Monday','Tuesday','Wednesday','Thursday','Friday']
    number_attacks = 10000
    number_benign = 10000
    attacks = []
    benign = []
    acount = 0
    bcount = 0
    
    
    while acount <= number_attacks or bcount <= number_benign:
        for day in days:
            df_attack,df_benign = select_data(day)
            attacks.append(df_attack)
            benign.append(df_benign)
            acount += len(df_attack.index)
            bcount += len(df_benign.index)
            
            
            
    
    
    
    pa = pd.concat(attacks)
    pb = pd.concat(benign)
    test_data = pd.concat([pa.iloc[:number_attacks],pb.iloc[:number_benign]])
    test_data.to_csv(f'FILENAME/CSV_data/TestData/RandomData_{i}_equal.csv', index=False)
    
    
for i in range(0,1):
    TestSelector(i)      
        
        
