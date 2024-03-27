# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 11:39:07 2024

@author: haydc
"""

import pandas as pd
import glob
import random
from math import floor
#Features = ['packets_AtoB','bytes_AtoB','AtoB_mean_pkt_len','AtoB_std_pkt_len','BtoA_std_pkt_len','AtoB_max_len','AtoB_avg_time','BtoA_avg_time','AtoB_time_std','BtoA_time_std','AtoB_max_inter_time','BtoA_max_inter_time','AtoB_mean_offset','AtoB_flag_RST','BtoA_flag_RST','AtoB_flag_PUSH','Attack_Type']
#Features = ['packets_AtoB','packets_BtoA','AtoB_avg_time','BtoA_avg_time','AtoB_time_std','BtoA_time_std','AtoB_min_inter_time','AtoB_max_inter_time','BtoA_min_inter_time','BtoA_max_inter_time','AtoB_mean_offset','AtoB_mean_pkt_len','AtoB_std_pkt_len','AtoB_min_len','AtoB_max_len','BtoA_mean_offset','BtoA_mean_pkt_len','BtoA_std_pkt_len','BtoA_min_len','BtoA_max_len','bytes_AtoB','bytes_BtoA','AtoB_flag_PUSH','AtoB_flag_RST','BtoA_flag_PUSH','BtoA_flag_RST','tot_std_bytes','ip_tl','tot_std_time','Attack','Attack_Type']
#Features = ['tot_std_bytes','BtoA_min_len','BtoA_mean_offset','BtoA_std_pkt_len','Total_TCP_flag_ACK','BtoA_mean_pkt_len','BtoA_min_inter_time','BtoA_max_inter_time','Total_TCP_flag_URG','BtoA_max_len','Total_TCP_flag_PUSH','Total_TCP_flag_RST','BtoA_flag_PUSH','BtoA_time_std','Total_TCP_flag_SYN','BtoA_avg_time','Attack','Attack_Type']
#Features = ['ip_tl','Total_TCP_flag_SYN','TCP_window','BtoA_max_inter_time','TCP_flags','ip_ttl','AtoB_max_inter_time','BtoA_std_pkt_len','BtoA_flag_RST','tot_flow_duration','tot_std_time','AtoB_mean_offset','AtoB_mean_pkt_len','Total_TCP_flag_FIN','bytes_BtoA','tot_bytes','Attack_Type']
#'tot_std_bytes','AtoB_avg_time','avg_time_btwn_pkts','bytes_ratio',
#Features = ['Total_TCP_flag_SYN','ip_tl','tot_std_bytes','TCP_window','BtoA_max_inter_time','TCP_flags','ip_ttl','AtoB_max_inter_time','tot_std_time','tot_bytes','BtoA_flag_RST','Total_TCP_flag_FIN','tot_flow_duration','bytes_BtoA','avg_time_btwn_pkts','BtoA_std_pkt_len','AtoB_mean_offset','Attack_Type']
#Features = ['AtoB_max_inter_time','BtoA_max_inter_time','ip_tl','TCP_window','tot_std_bytes','tot_flow_duration','BtoA_avg_time','AtoB_mean_offset','TCP_flags','AtoB_std_pkt_len','AtoB_time_std','tot_std_time','AtoB_max_len','BtoA_mean_pkt_len','BtoA_std_pkt_len','AtoB_mean_pkt_len','bytes_ratio','AtoB_avg_time','Total_TCP_flag_ACK','Total_TCP_flag_SYN','BtoA_min_inter_time','Attack_Type']
#Features = ['AtoB_max_inter_time','BtoA_max_inter_time','ip_tl','TCP_window','tot_std_bytes','tot_flow_duration','BtoA_avg_time','AtoB_mean_offset','TCP_flags','AtoB_std_pkt_len','AtoB_time_std','tot_std_time','AtoB_max_len','BtoA_mean_pkt_len','BtoA_std_pkt_len','AtoB_mean_pkt_len','Attack_Type']
Features = ['AtoB_max_inter_time','BtoA_max_inter_time','ip_tl','TCP_window','tot_std_bytes','tot_flow_duration','BtoA_avg_time','AtoB_mean_offset','TCP_flags','AtoB_std_pkt_len','AtoB_time_std','tot_std_time','AtoB_max_len','BtoA_mean_pkt_len','BtoA_std_pkt_len','AtoB_mean_pkt_len','bytes_ratio','Attack_Type']


def select_data(day):
    files = glob.glob(f'FILE_NAME/*.csv')
    fileLen = len(files)
    file_no = random.randint(0, fileLen - 1)
    data_no = random.randint(1, 400000 - 1000)
    df = pd.read_csv(f'FILE_NAME/{day}_flows_{file_no}.csv', skiprows=range(1, data_no), nrows=1000)
    df_benign = df[df['Attack'] == False]
    print(f'{day} flows {file_no}')
    del df
    return df_benign

def findData(day,noAttack,count,attackName):
    # print(count, f'{attackName}')
    df = pd.read_csv(f'FILE_NAME/Attacks/{attackName}_flows.csv')
    if(noAttack<len(df.index)):
        rand = random.randint(0,len(df.index)-noAttack)
        df_final = df.iloc[rand:rand+noAttack]
    else:
        df_final = df
    count1 = len(df_final.index)
    print(count1, f'{attackName}')
    return df_final, count1

def read_attack(count, no_attacks, day, attack_type, attacks):
    df = pd.read_csv(f'FILE_NAME/Attacks/{attack_type}_flows.csv')
    df = df[Features]
    df.replace({True: 1, False: 0}, inplace=True)
    df.fillna(0, inplace=True)
    df.drop_duplicates(ignore_index=True, inplace=True)
    attacks.append(df)

def generate_attack_data(count, no_attacks, day, attack_type, attacks, limit=30):
    df = pd.read_csv(f'FILE_NAME/Attacks/{attack_type}_flows.csv')
    df = df[Features]
    df.replace({True: 1, False: 0}, inplace=True)
    df.fillna(0, inplace=True)
    df.drop_duplicates(ignore_index=True, inplace=True)
    no = min(no_attacks,len(df))
    attacks.append(df.sample(no))
        
    #     if c < limit:
    #         print(count, attack_type)
    #         df_attack, _ = select_data(day)
    #         specific_attack = df_attack[df_attack['Attack_Type'] == attack_type]
    #         temp.append(specific_attack)
    #         count += len(specific_attack.index)
    #         c += 1
    #     else:
    #         df_attack, count1 = findData(day, no_attacks, count, attack_type)
    #         temp.append(df_attack)
    #         count += len(df_attack)
    #         print(count, f'{attack_type}')
    # attacks.append(pd.concat(temp, ignore_index=True).iloc[:no_attacks])

def TestSelector(i):
    days = ['Monday', 'Tuesday', 'Wednesday','Thursday','Friday']
    # , 'Thursday', 'Friday'
    no_benign = 5000
    no_ddos = 0
    no_port_scan = 0
    no_botnet = 0
    no_infiltration = 0
    no_web_attack_brute_force = 0
    no_web_attack_xss = 0
    no_dos_hulk = 160
    no_dos_golden_eye = 0
    no_dos_slowloris = 160
    no_dos_slowhttptest = 0
    no_ssh_patator = 0
    no_ftp_patator = 160

    count_benign = 0
    count_ddos = 0
    count_port_scan = 0
    count_botnet = 0
    count_infiltration = 0
    count_web_attack_brute_force = 0
    count_web_attack_xss = 0
    count_dos_hulk = 0
    count_dos_golden_eye = 0
    count_dos_slowloris = 0
    count_dos_slowhttptest = 0
    count_ssh_patator = 0
    count_ftp_patator = 0

    attacks = []
    benign = []

    # Benign
    while count_benign < no_benign:
        print(count_benign, ' Benign')
        for day in days:
            df_benign = select_data(day)
            # print(df_benign.columns)
            df_benign = df_benign[Features]
            df_benign.replace({True: 1, False: 0}, inplace=True)
            df_benign.fillna(0, inplace=True)
            df_benign.drop_duplicates(ignore_index=True, inplace=True)
            benign.append(df_benign)
            count_benign += len(df_benign.index)

    read_attack(count_ssh_patator, no_ssh_patator, 'Tuesday', 'SSH-Patator', attacks)
    read_attack(count_ftp_patator, no_ftp_patator, 'Tuesday', 'FTP-Patator', attacks)
    read_attack(count_dos_hulk, no_dos_hulk, 'Wednesday', 'DoS Hulk', attacks)
    # generate_attack_data(count_dos_golden_eye, no_dos_golden_eye, 'Wednesday', 'DoS GoldenEye', attacks)
    read_attack(count_dos_slowloris, no_dos_slowloris, 'Wednesday', 'DoS slowloris', attacks)
    # generate_attack_data(count_dos_slowhttptest, no_dos_slowhttptest, 'Wednesday', 'DoS Slowhttptest', attacks)
    # generate_attack_data(count_infiltration, no_infiltration, 'Thursday', 'Infiltration', attacks)
    # generate_attack_data(count_web_attack_brute_force, no_web_attack_brute_force, 'Thursday', 'Web Attack Brute Force', attacks)
    # generate_attack_data(count_web_attack_xss, no_web_attack_xss, 'Thursday', 'Web Attack XSS', attacks)
    # generate_attack_data(count_ddos, no_ddos, 'Friday', 'DDos', attacks)
    # generate_attack_data(count_port_scan, no_port_scan, 'Friday', 'PortScan', attacks)
    # generate_attack_data(count_botnet, no_botnet, 'Friday', 'Botnet', attacks)
    
    

    pa = pd.concat(attacks)
    pb = pd.concat(benign)
    pb = pb[Features]
    pb.replace({True: 1, False: 0}, inplace=True)
    pb.fillna(0, inplace=True)
    pb.drop_duplicates(ignore_index=True, inplace=True)
    test_data = pd.concat([pa, pb.iloc[:min(no_benign,len(pb.index))]])
    test_data.to_csv(f'FILE_NAME/TestDataZero/All_new{i}.csv', index=False)
    

for i in range(10, 11):
    TestSelector(i)
