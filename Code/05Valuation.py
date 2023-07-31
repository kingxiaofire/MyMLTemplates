from sklearn.metrics import classification_report
y_predict=model.predict(test_data)
y_pred_bool = np.argmax(y_predict, axis=1)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred_bool)

print(classification_report(y_test, y_pred_bool, digits=6))
print(cm)

evaluation = model.evaluate(test_data, y_test)

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