# -*- coding: utf-8 -*-
"""
Created on Tue 9/1/24

@author: haydc
"""

import pandas as pd
import dpkt
from dpkt.utils import inet_to_str
from datetime import datetime
from pytz import timezone
from dateutil import tz
import csv


def attack_tue(dic):
    # if no == 0:
    #     ret_val = False
    # else:
    #     ret_val = 'Benign'
    if(datetime(2017,7,4,9,20,0,tzinfo=timeshift)<= dic['timestamp'] <= datetime(2017,7,4,10,20,0,tzinfo=timeshift)):
        if(dic['ip_sourceAdd'] == '192.168.10.50' or dic['ip_destAdd'] == '192.168.10.50'):
            if(dic['ip_sourceAdd'] == '172.16.0.1' or dic['ip_destAdd'] == '172.16.0.1'):
                data_dic['Attack'] = True
                data_dic['Attack_Type'] = 'FTP-Patator'
            else:
                data_dic['Attack'] = False
                data_dic['Attack_Type'] = 'Benign'
        else:
            data_dic['Attack'] = False
            data_dic['Attack_Type'] = 'Benign'
       
    elif(datetime(2017,7,4,14,0,0,tzinfo=timeshift)<= dic['timestamp'] <= datetime(2017,7,4,15,0,0,tzinfo=timeshift)):
        if(dic['ip_sourceAdd'] == '192.168.10.50' or dic['ip_destAdd'] == '192.168.10.50'):
            if(dic['ip_sourceAdd'] == '172.16.0.1' or dic['ip_destAdd'] == '172.16.0.1'):
                data_dic['Attack'] = True
                data_dic['Attack_Type'] = 'SSH-Patator'
            else:
                data_dic['Attack'] = False
                data_dic['Attack_Type'] = 'Benign'
        else:
            data_dic['Attack'] = False
            data_dic['Attack_Type'] = 'Benign'
    else:
        data_dic['Attack'] = False
        data_dic['Attack_Type'] = 'Benign'
    
def attack_wed(dic):
    if(datetime(2017,7,5,9,47,0,tzinfo=timeshift)<= dic['timestamp'] <= datetime(2017,7,5,10,10,0,tzinfo=timeshift)):
        if(dic['ip_sourceAdd'] == '192.168.10.50' or dic['ip_destAdd'] == '192.168.10.50'):
            if(dic['ip_sourceAdd'] == '172.16.0.1' or dic['ip_destAdd'] == '172.16.0.1'):
                data_dic['Attack'] = True
                data_dic['Attack_Type'] = 'DoS slowloris'
            else:
                data_dic['Attack'] = False
                data_dic['Attack_Type'] = 'Benign'
        else:
            data_dic['Attack'] = False
            data_dic['Attack_Type'] = 'Benign'
        
    elif(datetime(2017,7,5,10,14,0,tzinfo=timeshift)<= dic['timestamp'] <= datetime(2017,7,5,10,35,0,tzinfo=timeshift)):
        if(dic['ip_sourceAdd'] == '192.168.10.50' or dic['ip_destAdd'] == '192.168.10.50'):
            if(dic['ip_sourceAdd'] == '172.16.0.1' or dic['ip_destAdd'] == '172.16.0.1'):
                data_dic['Attack'] = True
                data_dic['Attack_Type'] = 'DoS Slowhttptest'
            else:
                data_dic['Attack'] = False
                data_dic['Attack_Type'] = 'Benign'
        else:
            data_dic['Attack'] = False
            data_dic['Attack_Type'] = 'Benign'
        
    elif(datetime(2017,7,5,10,43,0,tzinfo=timeshift)<= dic['timestamp'] <= datetime(2017,7,5,11,0,0,tzinfo=timeshift)):
        if(dic['ip_sourceAdd'] == '192.168.10.50' or dic['ip_destAdd'] == '192.168.10.50'):
            if(dic['ip_sourceAdd'] == '172.16.0.1' or dic['ip_destAdd'] == '172.16.0.1'):
                data_dic['Attack'] = True
                data_dic['Attack_Type'] = 'DoS Hulk'
            else:
                data_dic['Attack'] = False
                data_dic['Attack_Type'] = 'Benign'
        else:
            data_dic['Attack'] = False
            data_dic['Attack_Type'] = 'Benign'
            
    elif(datetime(2017,7,5,11,10,0,tzinfo=timeshift)<= dic['timestamp'] <= datetime(2017,7,5,11,23,0,tzinfo=timeshift)):
        if(dic['ip_sourceAdd'] == '192.168.10.50' or dic['ip_destAdd'] == '192.168.10.50'):
            if(dic['ip_sourceAdd'] == '172.16.0.1' or dic['ip_destAdd'] == '172.16.0.1'):
                data_dic['Attack'] = True
                data_dic['Attack_Type'] = 'DoS GoldenEye'
            else:
                data_dic['Attack'] = False
                data_dic['Attack_Type'] = 'Benign'
        else:
            data_dic['Attack'] = False
            data_dic['Attack_Type'] = 'Benign'
        
    elif(datetime(2017,7,5,15,12,0,tzinfo=timeshift)<= dic['timestamp'] <= datetime(2017,7,5,15,32,0,tzinfo=timeshift)):
        print('heartbleed time')
        if(dic['ip_sourceAdd'] == '192.168.10.51' or dic['ip_destAdd'] == '192.168.10.51'):
            print('heartbleed address 1')
            if(dic['ip_sourceAdd'] == '172.16.0.1' or dic['ip_destAdd'] == '172.16.0.1'):
                print('heartbleed')
                data_dic['Attack'] = True
                data_dic['Attack_Type'] = 'Heartbleed'
            else:
                data_dic['Attack'] = False
                data_dic['Attack_Type'] = 'Benign'
        else:
            data_dic['Attack'] = False
            data_dic['Attack_Type'] = 'Benign'
    
    else:
        data_dic['Attack'] = False
        data_dic['Attack_Type'] = 'Benign'

def attack_thu(dic):
    # if no == 0:
    #     ret_val = False
    # else:
    #     ret_val = 'Benign'
    if(datetime(2017,7,6,9,20,0,tzinfo=timeshift)<= dic['timestamp'] <= datetime(2017,7,6,10,0,0,tzinfo=timeshift)):
        if(dic['ip_sourceAdd'] == '192.168.10.50' or dic['ip_destAdd'] == '192.168.10.50'):
            if(dic['ip_sourceAdd'] == '172.16.0.1' or dic['ip_destAdd'] == '172.16.0.1'):
                data_dic['Attack'] = True
                data_dic['Attack_Type'] = 'Web Attack Brute Force'
            else:
                data_dic['Attack'] = False
                data_dic['Attack_Type'] = 'Benign'
        else:
            data_dic['Attack'] = False
            data_dic['Attack_Type'] = 'Benign'
        
    elif(datetime(2017,7,6,10,15,0,tzinfo=timeshift)<= dic['timestamp'] <= datetime(2017,7,6,10,35,0,tzinfo=timeshift)):
        if(dic['ip_sourceAdd'] == '192.168.10.50' or dic['ip_destAdd'] == '192.168.10.50'):
            if(dic['ip_sourceAdd'] == '172.16.0.1' or dic['ip_destAdd'] == '172.16.0.1'):
                data_dic['Attack'] = True
                data_dic['Attack_Type'] = 'Web Attack XSS'
            else:
                data_dic['Attack'] = False
                data_dic['Attack_Type'] = 'Benign'
        else:
            data_dic['Attack'] = False
            data_dic['Attack_Type'] = 'Benign'
        
    elif(datetime(2017,7,6,10,40,0,tzinfo=timeshift)<= dic['timestamp'] <= datetime(2017,7,6,10,42,0,tzinfo=timeshift)):
        if(dic['ip_sourceAdd'] == '192.168.10.50' or dic['ip_destAdd'] == '192.168.10.50'):
            if(dic['ip_sourceAdd'] == '172.16.0.1' or dic['ip_destAdd'] == '172.16.0.1'):
                data_dic['Attack'] = True
                data_dic['Attack_Type'] = 'Web Attack Sql Injection'
            else:
                data_dic['Attack'] = False
                data_dic['Attack_Type'] = 'Benign'
        else:
            data_dic['Attack'] = False
            data_dic['Attack_Type'] = 'Benign'
        
    elif(datetime(2017,7,6,14,19,0,tzinfo=timeshift)<= dic['timestamp'] <= datetime(2017,7,6,14,21,0,tzinfo=timeshift)):
        if(dic['ip_sourceAdd'] == '192.168.10.8' or dic['ip_destAdd'] == '192.168.10.8'):
            if(dic['ip_sourceAdd'] == '205.174.165.73' or dic['ip_destAdd'] == '205.174.165.73'):
                data_dic['Attack'] = True
                data_dic['Attack_Type'] = 'Infiltration'
            else:
                data_dic['Attack'] = False
                data_dic['Attack_Type'] = 'Benign'
        else:
            data_dic['Attack'] = False
            data_dic['Attack_Type'] = 'Benign'
        
    elif(datetime(2017,7,6,14,33,0,tzinfo=timeshift)<= dic['timestamp'] <= datetime(2017,7,6,14,35,0,tzinfo=timeshift)):
        if(dic['ip_sourceAdd'] == '192.168.10.8' or dic['ip_destAdd'] == '192.168.10.8'):
            if(dic['ip_sourceAdd'] == '205.174.165.73' or dic['ip_destAdd'] == '205.174.165.73'):
                data_dic['Attack'] = True
                data_dic['Attack_Type'] = 'Infiltration'
            else:
                data_dic['Attack'] = False
                data_dic['Attack_Type'] = 'Benign'
        else:
            data_dic['Attack'] = False
            data_dic['Attack_Type'] = 'Benign'
    
    elif(datetime(2017,7,6,14,53,0,tzinfo=timeshift)<= dic['timestamp'] <= datetime(2017,7,6,15,0,0,tzinfo=timeshift)):
        if(dic['ip_sourceAdd'] == '192.168.10.25' or dic['ip_destAdd'] == '192.168.10.25'):
            if(dic['ip_sourceAdd'] == '205.174.165.73' or dic['ip_destAdd'] == '205.174.165.73'):
                data_dic['Attack'] = True
                data_dic['Attack_Type'] = 'Infiltration'
            else:
                data_dic['Attack'] = False
                data_dic['Attack_Type'] = 'Benign'
        else:
            data_dic['Attack'] = False
            data_dic['Attack_Type'] = 'Benign'
    
    elif(datetime(2017,7,6,15,4,0,tzinfo=timeshift)<= dic['timestamp'] <= datetime(2017,7,6,15,45,0,tzinfo=timeshift)):
        if(dic['ip_sourceAdd'] == '192.168.10.8' or dic['ip_destAdd'] == '192.168.10.8'):
            data_dic['Attack'] = True
            data_dic['Attack_Type'] = 'Infiltration'
        else:
            data_dic['Attack'] = False
            data_dic['Attack_Type'] = 'Benign'
    else:
        data_dic['Attack'] = False
        data_dic['Attack_Type'] = 'Benign'
    
def attack_fri(dic):
    if(datetime(2017,7,7,10,2,0,tzinfo=timeshift)<= dic['timestamp'] <= datetime(2017,7,7,11,2,0,tzinfo=timeshift)):
        if(dic['ip_sourceAdd'] == '205.174.165.73' or dic['ip_destAdd'] == '205.174.165.73'):
            if(dic['ip_sourceAdd'] == '192.168.10.15' or dic['ip_destAdd'] == '192.168.10.15'):
                data_dic['Attack'] = True
                data_dic['Attack_Type'] = 'Botnet'
            elif(dic['ip_sourceAdd'] == '192.168.10.9' or dic['ip_destAdd'] == '192.168.10.9'):
                data_dic['Attack'] = True
                data_dic['Attack_Type'] = 'Botnet'
            elif(dic['ip_sourceAdd'] == '192.168.10.14' or dic['ip_destAdd'] == '192.168.10.14'):
                data_dic['Attack'] = True
                data_dic['Attack_Type'] = 'Botnet'
            elif(dic['ip_sourceAdd'] == '192.168.10.5' or dic['ip_destAdd'] == '192.168.10.5'):
                data_dic['Attack'] = True
                data_dic['Attack_Type'] = 'Botnet'
            elif(dic['ip_sourceAdd'] == '192.168.10.8' or dic['ip_destAdd'] == '192.168.10.8'):
                data_dic['Attack'] = True
                data_dic['Attack_Type'] = 'Botnet'
            else:
                data_dic['Attack'] = False
                data_dic['Attack_Type'] = 'Benign'
        else:
            data_dic['Attack'] = False
            data_dic['Attack_Type'] = 'Benign'
    
        
    elif(datetime(2017,7,7,14,51,0,tzinfo=timeshift)<= dic['timestamp'] <= datetime(2017,7,7,15,29,0,tzinfo=timeshift)):
        if(dic['ip_sourceAdd'] == '192.168.10.50' or dic['ip_destAdd'] == '192.168.10.50'):
            if(dic['ip_sourceAdd'] == '172.16.0.1' or dic['ip_destAdd'] == '172.16.0.1'):
                data_dic['Attack'] = True
                data_dic['Attack_Type'] = 'PortScan'
            else:
                data_dic['Attack'] = False
                data_dic['Attack_Type'] = 'Benign'
        else:
            data_dic['Attack'] = False
            data_dic['Attack_Type'] = 'Benign'
            
    elif(datetime(2017,7,7,15,56,0,tzinfo=timeshift)<= dic['timestamp'] <= datetime(2017,7,7,16,16,0,tzinfo=timeshift)):
        if(dic['ip_sourceAdd'] == '192.168.10.50' or dic['ip_destAdd'] == '192.168.10.50'):
            if(dic['ip_sourceAdd'] == '172.16.0.1' or dic['ip_destAdd'] == '172.16.0.1'):
                data_dic['Attack'] = True
                data_dic['Attack_Type'] = 'DDos'
            else:
                data_dic['Attack'] = False
                data_dic['Attack_Type'] = 'Benign'
        else:
            data_dic['Attack'] = False
            data_dic['Attack_Type'] = 'Benign'
    else:
        data_dic['Attack'] = False
        data_dic['Attack_Type'] = 'Benign'
        

def label_attacks(dic,day):    
    if(day=='Monday'):
        data_dic['Attack'] = False
        data_dic['Attack_Type'] = 'Benign'
    elif(day=='Tuesday'):
        attack_tue(dic)
    elif(day=='Wednesday'):
        attack_wed(dic)
    elif(day=='Thursday'):
        attack_thu(dic)
    elif(day=='Friday'):
        attack_fri(dic)        
        
    

timeshift = tz.gettz('Canada/Atlantic')

days = ['Wednesday']
for day in days:
    with open(f'Filename/{day}-WorkingHours.pcap',"rb") as f:
        with open(f'Filename/Multiclass/LabelledCSV/{day}.csv','w',newline='') as f1:
            pcap = dpkt.pcapng.Reader(f)
            count = 0
            skip_count = 0
            data_dic = {
                    'timestamp': [],
                    'ip_v' : [],
                    'ip_hl': [],
                    'ip_tos': [],
                    'ip_tl': [],
                    'ip_id': [],
                    'ip_flag_rf': [],
                    'ip_flag_df': [],
                    'ip_flag_mf': [],
                    'ip_fragoff': [],
                    'ip_ttl': [],
                    'ip_protocol': [], 
                    'ip_hchecksum': [],
                    'ip_sourceAdd': [],
                    'ip_destAdd': [],
                    'ip_options':[],
                    'source_port': [],
                    'destination_port': [],
                    'TCP_Sequence_no': [],
                    'TCP_akn_no': [],
                    'TCP_DO': [],
                    'TCP_flags': [],
                    'TCP_flag_FIN': [],
                    'TCP_flag_SYN': [],
                    'TCP_flag_RST': [],
                    'TCP_flag_PUSH': [],
                    'TCP_flag_ACK': [],
                    'TCP_flag_URG': [],
                    'TCP_flag_ECE': [],
                    'TCP_flag_CWR': [],
                    'TCP_window': [],
                    'Checksum': [],
                    'TCP_urgent_pointer': [],
                    'TCP_Option': [],
                    'UDP_length': [],      
                    'Attack': [],
                    'Attack_Type': [],
            }

            doc_count = 0
            w = csv.DictWriter(f1, data_dic.keys())
            w.writeheader()
            for timestamp, buffer in pcap:
                # if (doc_count >= 5000000):
                #       break
                eth = dpkt.ethernet.Ethernet(buffer)
                ip1 = eth.data
                
                
                if not isinstance(ip1, dpkt.ip.IP):
                    skip_count += 1
                    continue 
                transport_layer = ip1.data
                if not isinstance(transport_layer,dpkt.tcp.TCP) or isinstance(transport_layer,dpkt.udp.UDP):
                    skip_count += 1
                    continue
                doc_count +=1
                
                data_dic['timestamp'] = datetime.fromtimestamp(timestamp,tz=timeshift )
                data_dic['ip_v'] = ip1.v 
                data_dic['ip_hl'] = ip1.hl 
                data_dic['ip_tos'] = ip1.tos 
                data_dic['ip_tl'] = ip1.len 
                data_dic['ip_id'] = ip1.id 
                data_dic['ip_flag_rf'] = ip1.rf 
                data_dic['ip_flag_df'] = ip1.df 
                data_dic['ip_flag_mf'] = ip1.mf 
                data_dic['ip_fragoff'] = ip1.offset #not sure about this 
                data_dic['ip_ttl'] = ip1.ttl 
                data_dic['ip_protocol'] = ip1.p  #categorical
                data_dic['ip_hchecksum'] = ip1.sum 
                data_dic['ip_sourceAdd'] = inet_to_str(ip1.src )
                data_dic['ip_destAdd'] = inet_to_str(ip1.dst )
                data_dic['ip_options'] = ip1.opts 
                
                
                if isinstance(transport_layer, dpkt.tcp.TCP):
                    data_dic['source_port'] = transport_layer.sport 
                    data_dic['destination_port'] = transport_layer.dport 
                    data_dic['TCP_Sequence_no'] = transport_layer.seq 
                    data_dic['TCP_akn_no'] = transport_layer.ack 
                    data_dic['TCP_DO'] = transport_layer.off 
                    data_dic['TCP_flags'] = transport_layer.flags 
                    
                    data_dic['TCP_flag_FIN']  = (transport_layer.flags & dpkt.tcp.TH_FIN ) != 0  # end of data
                    data_dic['TCP_flag_SYN'] = (transport_layer.flags & dpkt.tcp.TH_SYN) !=0   # synchronize sequence numbers
                    data_dic['TCP_flag_RST'] = (transport_layer.flags & dpkt.tcp.TH_RST ) !=0   # reset connection
                    data_dic['TCP_flag_PUSH'] = (transport_layer.flags & dpkt.tcp.TH_PUSH ) !=0  # push
                    data_dic['TCP_flag_ACK'] = (transport_layer.flags & dpkt.tcp.TH_ACK ) !=0  # acknowledgment number set
                    data_dic['TCP_flag_URG'] = (transport_layer.flags & dpkt.tcp.TH_URG ) !=0  # urgent pointer set
                    data_dic['TCP_flag_ECE'] = (transport_layer.flags & dpkt.tcp.TH_ECE ) !=0  # ECN echo, RFC 3168
                    data_dic['TCP_flag_CWR'] = (transport_layer.flags & dpkt.tcp.TH_CWR ) !=0  # congestion window reduced
                    
                    data_dic['TCP_window'] = transport_layer.win 
                    data_dic['Checksum'] = transport_layer.sum 
                    data_dic['TCP_urgent_pointer'] = transport_layer.urp 
                    data_dic['TCP_Option'] = transport_layer.opts 
                    data_dic['UDP_length'] = 0 
                    
                    
                    
                    
                elif isinstance(transport_layer, dpkt.udp.UDP):
                    data_dic['source_port'] = transport_layer.sport 
                    data_dic['destination_port'] = transport_layer.dport 
                    data_dic['TCP_Sequence_no'] = 0 
                    data_dic['TCP_akn_no'] = 0 
                    data_dic['TCP_DO'] = 0 
                    data_dic['TCP_flags'] = 0 
                    data_dic['TCP_window'] = 0 
                    data_dic['Checksum'] = 0 
                    data_dic['TCP_urgent_pointer'] = 0 
                    data_dic['TCP_Option'] = 0 
                    data_dic['UDP_length'] = transport_layer.ulen 
                    
                    data_dic['TCP_flag_FIN']  = 0  # N/a to UDP
                    data_dic['TCP_flag_SYN'] = 0   # N/a to UDP
                    data_dic['TCP_flag_RST'] = 0   # N/a to UDP
                    data_dic['TCP_flag_PUSH'] = 0  # N/a to UDP
                    data_dic['TCP_flag_ACK'] = 0  # N/a to UDP
                    data_dic['TCP_flag_URG'] = 0  # N/a to UDP
                    data_dic['TCP_flag_ECE'] = 0  # N/a to UDP
                    data_dic['TCP_flag_CWR'] = 0  # N/a to UDP
                    
                # elif isinstance(transport_layer, dpkt.icmp.ICMP):
                #     protocol_type = "ICMP"
                    
                else:
                    #print('skipped' + str(count))
                    continue
                if (doc_count % 10000 == 0):
                    print(doc_count)
                
                label_attacks(data_dic,day)
                
                # if(attack):
                #     print(attack)
                #     print(data_dic['Attack'])
                
                
                # data_dic['Attack_Type'] = label_attacks(data_dic,day,1,'Benign')
                # if(data_dic['Attack'] == True):    
                #     print("in here")
                    
                #     data_dic['Attack_Type'] = 
                # else:
                #     data_dic['Attack_Type'] = 'Benign'
                w.writerow(data_dic);
    
    f.close()
    f1.close()
    if(f.closed):
        print('complete')
    else:
        f.close()

    if(f1.closed):
        print('complete')
    else:
        f1.close()








    # if(data_dic['ip_sourceAdd'] == '205.174.165.73'):
    #     break
        
        
    
    
    

    
    
#df = pd.DataFrame(data_dic)

    
    
    


