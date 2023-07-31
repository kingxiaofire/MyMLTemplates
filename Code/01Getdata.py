# -*- coding:utf-8 -*-
import pandas as pd
import os

df = pd.read_csv('/MachineLearningCVE/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv', skipinitialspace=True )
print(df.info())

def _renaming_class_label(df: pd.DataFrame):
    # 去除不可识别的非utf-8编码字符
    labels = {"Web Attack � Brute Force": "Web_Attack_Brute_Force"
              ,"Web Attack � XSS": "Web_Attack_XSS"
              ,"Web Attack � Sql Injection": "Web_Attack_Sql_Injection"
             }

    for old_label, new_label in labels.items():
        df.Label.replace(old_label, new_label, inplace=True)

# Renaming labels
_renaming_class_label(df)

# Save to csv
df.to_csv('MachineLearningCVE/00_Thursday-WorkingHours-Morning-WebAttacks_handle.pcap_ISCX.csv', index=False)

# Combine all data to one CSV file
DIR_PATH = "MachineLearningCVE/"
FILE_NAMES = ["Monday-WorkingHours.pcap_ISCX.csv"
              ,"Tuesday-WorkingHours.pcap_ISCX.csv"
              ,"Wednesday-workingHours.pcap_ISCX.csv"
              ,"00_Thursday-WorkingHours-Morning-WebAttacks_handle.pcap_ISCX.csv"
              ,"Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv"
              ,"Friday-WorkingHours-Morning.pcap_ISCX.csv"
              ,"Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv"
              ,"Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
             ]


df = [pd.read_csv(os.path.join(DIR_PATH, f), skipinitialspace=True) for f in FILE_NAMES]
df = pd.concat(df, ignore_index=True)
print(df.info())

PROCESSED_DIR_PATH = "MachineLearningCVE/"
df.to_csv(os.path.join(PROCESSED_DIR_PATH, "All_Of_MachineLearningCVE.csv"), index=False)