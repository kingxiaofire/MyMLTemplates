import tensorflow as tf
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"  

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
##降低维度后...
print(train_data.shape)
print(X_test.shape)
""" 
(32000, 32, 1)
(8000, 69)
"""
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.layers import Dense, Flatten, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.metrics import Precision, Recall


# 定义 ResNet50 模型
resnet = ResNet50(input_shape=(32, 32, 1), weights=None, classes=4)
# 添加 Reshape 层
x = resnet.layers[-2].output
x = Reshape((1, 1, 2048))(x)

# 添加额外的全局平均池化层和分类层
x = Flatten()(x)
x = Dense(4, activation='softmax')(x)

# 创建新的模型
model = Model(inputs=resnet.input, outputs=x)

# 编译模型
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
#model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy', Precision(), Recall()])

MyModel_history = model.fit(train_data, y_train, epochs=200, batch_size=64, validation_data=(test_data, y_test))

from sklearn.metrics import classification_report
y_predict=model.predict(test_data)
y_pred_bool = np.argmax(y_predict, axis=1)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred_bool)

print(classification_report(y_test, y_pred_bool, digits=6))
print(cm)

evaluation = model.evaluate(test_data, y_test)

loss, accuracy, precision, recall = evaluation[0], evaluation[1], evaluation[2], evaluation[3]

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
accuracy

val_accuracy = MyModel_history.history['val_accuracy']
val_accuracy

loss = MyModel_history.history['loss']
loss

val_loss = MyModel_history.history['val_loss']
val_loss

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
confusion = cm
# 热度图，后面是指定的颜色块，可设置其他的不同颜色、
fig = plt.figure(figsize=(8, 8), dpi=400)
plt.imshow(confusion, cmap=plt.cm.Blues)
# ticks 坐标轴的坐标点
# label 坐标轴标签说明
indices = range(len(confusion))
# plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.rcParams['axes.unicode_minus'] = False
# # 第一个是迭代对象，表示坐标的显示顺序，第二个参数是坐标轴显示列表
plt.xticks(indices, ['BENGIN','DDoS','DoS','Patator'],size = 14)
plt.yticks(indices, ['BENGIN','DDoS','DoS','Patator'],size = 14)

plt.colorbar(fraction=0.0455, pad=0.05)


# plt.rcParams两行是用于解决标签不能显示汉字的问题
#plt.rcParams['font.sans-serif']=['SimHei']
#plt.rcParams['axes.unicode_minus'] = False
plt.xlabel('True Label',fontsize = 16)
plt.ylabel('Predicted Label',fontsize = 16)
#plt.title('lightGBM',fontsize = 16)

# 显示数据
for first_index in range(len(confusion)):    #第几行
    for second_index in range(len(confusion[first_index])):    #第几列
        plt.text(first_index, second_index, confusion[first_index][second_index],ha="center",color = 'g',size = 18)
# 在matlab里面可以对矩阵直接imagesc(confusion)
# 显示
#plt.savefig("./XGBMatrix.png",bbox_inches='tight')
plt.show()