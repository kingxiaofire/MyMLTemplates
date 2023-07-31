import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('MachineLearningCVE/01_All_Of_MachineLearningCVE.csv', skipinitialspace=True )

## 标签组合
def attack_classify(tag):
    dic_attack_type ={
         'BENIGN':['BENIGN']
        ,'DOS':['DoS Hulk','DoS GoldenEye','DoS slowloris','DoS Slowhttptest','DDoS','Heartbleed']
        ,'PortScan':['PortScan']
        ,'Brute Force':['FTP-Patator','SSH-Patator']
        ,'Web Attack':['Web_Attack_Brute_Force','Web_Attack_XSS','Web_Attack_Sql_Injection']
        ,'Bot':['Bot']
        ,'Infiltration':['Infiltration']
    }
    for i in dic_attack_type.keys():
        if tag in dic_attack_type[i]:
            return i
    else:
        return tag
    
df['target_type']=df.Label.apply(attack_classify)
df.target_type.value_counts()


labelencoder = LabelEncoder()
df.iloc[:, -1] = labelencoder.fit_transform(df.iloc[:, -1])
print(df.target_type.value_counts())

#e.g. df.drop(index=(df.loc[(df['course']=='test')&(df['table']<'4')].index),inplace=True)
df.drop(index=(df.loc[(df['target_type']==4)].index),inplace=True)

df.drop(['Fwd Header Length.1'## 34  Fwd_Header_Length 和 55 Fwd_Header_Length.1重复
        ,'Bwd PSH Flags'
        ,'Bwd URG Flags'
        ,'Fwd Avg Bytes/Bulk'
        ,'Fwd Avg Packets/Bulk'
        ,'Fwd Avg Bulk Rate'
        ,'Bwd Avg Bytes/Bulk'
        ,'Bwd Avg Packets/Bulk'
        ,'Bwd Avg Bulk Rate'
        #,'Flow Bytes/s'      ##  Flow_Bytes_P_s 中NaN太多了 
        #,'Flow Packets/s'    ## 有异常
        ]
        ,axis=1, inplace=True) ## 流量持续时间为0的记录，以及具有冗余信息的记录，删除 ;

NaNumFB1 = df['Flow Bytes/s'].isnull().sum() 
print('------------ Flow Bytes/s -------------')
print("原先有NaN的个数为：",NaNumFB1)

df['Flow Bytes/s'] = df['Flow Bytes/s'].replace([np.inf, -np.inf], np.nan)
NaNumFB2 = df['Flow Bytes/s'].isnull().sum() 
print("将 inf 换成NaN，现在NaN有：",NaNumFB2)

df['Flow Bytes/s'].fillna(df['Flow Bytes/s'].mean(),inplace=True)
NaNumFB3 = df['Flow Bytes/s'].isnull().sum() 
print("用均值替换NaN,现在NaN有：",NaNumFB3)




NaNFP1 = df['Flow Packets/s'].isnull().sum() 
print('------------ Flow Packets/s -------------')
print("原先有NaN的个数为：",NaNFP1)

df['Flow Packets/s'] = df['Flow Packets/s'].replace([np.inf, -np.inf], np.nan)
NaNFP2 = df['Flow Packets/s'].isnull().sum() 
print("将 inf 换成NaN，现在NaN有：",NaNFP2)

df['Flow Packets/s'].fillna(df['Flow Packets/s'].mean(),inplace=True)
NaNFP3 = df['Flow Packets/s'].isnull().sum() 
print("用均值替换NaN,现在NaN有：",NaNFP3)

df.to_csv("MachineLearningCVE/03_handled_data", index=False)