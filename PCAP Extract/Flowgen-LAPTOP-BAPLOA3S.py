# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 22:32:43 2024

@author: haydc
"""

#Flow Generator

import pandas as pd



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

def flow_gen(day):
        
    df = pd.read_csv(f'Filename/LabelledCSV/{day}.csv',dtype=dtypes,parse_dates=['timestamp'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], format="mixed")
    print(f'{day} opened')
    
    df['flow_id'] = df.apply(lambda row: ','.join(sorted([row['ip_sourceAdd'], row['ip_destAdd'], str(row['source_port']), str(row['destination_port']), str(row['ip_protocol'])])), axis=1)
    
    
    grouped_flows = df.groupby('flow_id')
    flows = []
    split_flows = []
    count = 0
    pkt_count = 0
    temp_count = 0
    for flow_id, flow_data in grouped_flows:
        # if(count>10):
        #     break   
        if( count % 1000 == 0):
            print(count)
        pkt_count += len(flow_data)
        flow_data['flow'] = count
        flow_data['flow_length'] = len(flow_data)
        flow_data['avg_pack_length'] = flow_data['flow_length'].mean()
        flow_data['avg_time_btwn_pkts'] = flow_data['timestamp'].diff().mean().total_seconds()
        start_time = flow_data['timestamp'].min()
        end_time = flow_data['timestamp'].max()
        flow_data['tot_flow_duration'] = (end_time-start_time).total_seconds()
        flow_data['avg_pkt_time'] = flow_data['tot_flow_duration']/flow_data['flow_length']
        flow_data['tot_bytes']=flow_data['ip_tl'].sum()
        flow_data['tot_std_bytes'] = flow_data['ip_tl'].std() 
        flow_data['tot_std_time'] = flow_data['timestamp'].diff().std().total_seconds()
        # flow_data['Total_ip_flag_rf'] = flow_data[flow_data['ip_flag_rf'] == 1]['ip_flag_rf'].sum()
        # flow_data['Total_ip_flag_df'] = flow_data[flow_data['ip_flag_df'] == 1]['ip_flag_df'].sum()
        # flow_data['Total_ip_flag_mf'] = flow_data[flow_data['ip_flag_mf'] == 1]['ip_flag_mf'].sum()
        flow_data['Total_TCP_flag_FIN'] = flow_data[flow_data['TCP_flag_FIN'] == True]['TCP_flag_FIN'].sum()
        flow_data['Total_TCP_flag_SYN'] = flow_data[flow_data['TCP_flag_SYN'] == True]['TCP_flag_SYN'].sum()
        flow_data['Total_TCP_flag_RST'] = flow_data[flow_data['TCP_flag_RST'] == True]['TCP_flag_RST'].sum()
        flow_data['Total_TCP_flag_PUSH'] = flow_data[flow_data['TCP_flag_PUSH'] == True]['TCP_flag_PUSH'].sum()
        flow_data['Total_TCP_flag_ACK'] = flow_data[flow_data['TCP_flag_ACK'] == True]['TCP_flag_ACK'].sum()
        flow_data['Total_TCP_flag_URG'] = flow_data[flow_data['TCP_flag_URG'] == True]['TCP_flag_URG'].sum()        		
        flow_data['Total_TCP_flag_ECE'] = flow_data[flow_data['TCP_flag_ECE'] == True]['TCP_flag_ECE'].sum()
        flow_data['Total_TCP_flag_CWR'] = flow_data[flow_data['TCP_flag_CWR'] == True]['TCP_flag_CWR'].sum()
        
        
        individual_flow = flow_data.groupby(['ip_sourceAdd'])
        in_count = 0
        temp_bytes = 0
        for [ind_ip_source], flow_direction in individual_flow:
            if in_count == 0:
                a = 'A'
                b = 'B'
                temp_bytes = flow_direction['ip_tl'].sum()
            elif in_count == 1:
                a = 'B'
                b = 'A'
                flow_data['bytes_ratio'] = temp_bytes/flow_direction['ip_tl'].sum() 
            elif in_count>1:
                print('this aint right')
                break
            in_count +=1
            flow_data[f'packets_{a}to{b}'] = len(flow_direction)
            flow_data[f'bytes_{a}to{b}'] = flow_direction['ip_tl'].sum()
            flow_data[f'{a}to{b}_avg_time'] = flow_direction['timestamp'].diff().mean().total_seconds()
            flow_data[f'{a}to{b}_time_std'] = flow_direction['timestamp'].diff().std().total_seconds()
            flow_data[f'{a}to{b}_min_inter_time'] = flow_direction['timestamp'].diff().min().total_seconds()
            flow_data[f'{a}to{b}_max_inter_time'] = flow_direction['timestamp'].diff().max().total_seconds()
            flow_data[f'{a}to{b}_mean_offset'] = flow_direction['TCP_DO'].mean()
            flow_data[f'{a}to{b}_mean_pkt_len'] = flow_direction['ip_tl'].mean()
            flow_data[f'{a}to{b}_std_pkt_len'] = flow_direction['ip_tl'].std()
            flow_data[f'{a}to{b}_min_len'] = flow_direction['ip_tl'].min()
            flow_data[f'{a}to{b}_max_len'] = flow_direction['ip_tl'].max()
            flow_data[f'{a}to{b}_flag_PUSH'] = flow_direction[flow_direction['TCP_flag_PUSH'] == True]['TCP_flag_PUSH'].sum()
            flow_data[f'{a}to{b}_flag_RST'] = flow_direction[flow_direction['TCP_flag_RST'] == True]['TCP_flag_RST'].sum()
            flow_data[f'{a}to{b}_flag_URG'] = flow_direction[flow_direction['TCP_flag_URG'] == True]['TCP_flag_URG'].sum()        		
            
        # flows.append(flow_data)
        split_flows.append(flow_data)
        if pkt_count >=400000:
            # output_filename = f'Filename/CSV_data/{day}/{day}_flows.csv'
            pd.concat(split_flows,ignore_index=False).to_csv(f'C:/Users/haydc/OneDrive - University of Strathclyde/Year4/19496/Dataset (1)/Multiclass/Flows/{day}/{day}_flows_{temp_count}.csv',index=False)
            
            temp_count += 1
            split_flows = []
            pkt_count = 0
        
        #print('---')
        count +=1
        
    del split_flows
    # output_filename = f'Filename/{day}/{day}_flows.csv'
    # pd.concat(flows, ignore_index=False).to_csv(output_filename, index=False)
    del flows
    print(f'complete {day}')
    print('total flows: ', count)
    

days = ['Thursday','Friday']
for day in days:
    print(day)
    flow_gen(day)
    
    


