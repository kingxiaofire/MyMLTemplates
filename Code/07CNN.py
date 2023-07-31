import tensorflow as tf
import os
from sklearn.model_selection import train_test_split 

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  
## 这是指定显卡的序号

import numpy as np
import pandas as pd
from sklearn.preprocessing import QuantileTransformer
from sklearn.metrics import classification_report
from sklearn.metrics import classification_report

ALLData = pd.read_csv('./ALLData.csv',skipinitialspace=True)
X = ALLData.iloc[:,:-2]
X = X.astype(float)
y = ALLData.iloc[:,-1]

# 将数据集分割为训练集和测试集，其中test_size为测试集所占的比例
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# 打印训练集和测试集的形状
print("训练集形状：", X_train.shape, y_train.shape)
print("测试集形状：", X_test.shape, y_test.shape)
""" 
训练集形状： (32000, 69) (32000,)
测试集形状： (8000, 69) (8000,)
"""
train_data = X_train.values.reshape(-1, 1, 69)
test_data = X_test.values.reshape(-1, 1, 69)
print(train_data.shape)
print(X_test.shape)
""" 
(32000, 1, 69)
(8000, 69)
"""

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

model_input = keras.Input(shape=(1,69), name="model_input")

cnn = layers.Convolution1D(8, 3, padding="same",activation="relu",name='cnn_layer1')(model_input)
cnn = layers.MaxPooling1D(pool_size=2,strides=1, padding='same',name='cnn_layer2')(cnn)
cnn = layers.Flatten(name='cnn_layer4')(cnn)

cnn = layers.Dense(64, activation="relu",name='cnn_layer5')(cnn)
cnn = layers.Dense(8, activation="sigmoid",name='cnn_layer7')(cnn)

model = keras.Model(inputs=[model_input],
                    outputs=[cnn],)
#print(model.summary())
#keras.utils.plot_model(model,show_shapes=True)
model.compile(loss="sparse_categorical_crossentropy", optimizer="adam",metrics=['accuracy'])

MyModel_history = model.fit(train_data, y_train, epochs=20, batch_size=64, validation_data=(test_data, y_test))


y_predict=model.predict(test_data)
y_pred_bool = np.argmax(y_predict, axis=1)
print(classification_report(y_test, y_pred_bool, digits=6))

import matplotlib.pyplot as plt
plt.plot(MyModel_history.history['loss'],label='train')# 训练集的loss
plt.plot(MyModel_history.history['val_loss'],label='val')
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend()
plt.show()

plt.plot(MyModel_history.history['accuracy'],label='train') # 训练集的acc
plt.plot(MyModel_history.history['val_accuracy'],label='val') 
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend()
plt.show()

accuracy = MyModel_history.history['accuracy']
print(accuracy)

val_accuracy = MyModel_history.history['val_accuracy']
print(val_accuracy)