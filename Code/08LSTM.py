import tensorflow as tf
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  

import numpy as np
import pandas as pd
from sklearn.preprocessing import QuantileTransformer

ALLData = pd.read_csv('./ALLData.csv',skipinitialspace=True)

X = ALLData.iloc[:,:-2]
X = X.astype(float)

y = ALLData.iloc[:,-1]

from sklearn.model_selection import train_test_split 

# 将数据集分割为训练集和测试集，其中test_size为测试集所占的比例
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 打印训练集和测试集的形状
print("训练集形状：", X_train.shape, y_train.shape)
print("测试集形状：", X_test.shape, y_test.shape)

train_data = X_train.values.reshape(-1, 1, 69)
test_data = X_test.values.reshape(-1, 1, 69)
print(train_data.shape)
print(X_test.shape)
""" 
训练集形状： (32000, 69) (32000,)
测试集形状： (8000, 69) (8000,)
(32000, 1, 69)
(8000, 69)
"""

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential([
    layers.LSTM(8, input_shape=(1, 69)),
    layers.Dense(4, activation='softmax')
])

model.compile(loss="sparse_categorical_crossentropy", optimizer="adam",metrics=['accuracy'])

MyModel_history = model.fit(train_data, y_train, epochs=20, batch_size=64, validation_data=(test_data, y_test))