import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score,precision_recall_fscore_support

start = time.time()
DT_score=DT.score(TestX,TestY)
end = time.time()
print(end-start)

start = time.time()
y_predict=DT.predict(TestX)
end = time.time()
print(end-start)

y_true=TestY
print('Accuracy of DT: '+ str(DT_score))
precision,recall,fscore,none= precision_recall_fscore_support(y_true, y_predict, average='weighted') 
print('Precision of DT: '+(str(precision)))
print('Recall of DT: '+(str(recall)))
print('F1-score of DT: '+(str(fscore)))
print(classification_report(y_true,y_predict,digits=6))


import matplotlib.pyplot as plt
import seaborn as sns
confusion = confusion_matrix(y_true,y_predict)
# 热度图，后面是指定的颜色块，可设置其他的不同颜色、
fig = plt.figure(figsize=(8, 8), dpi=100)
plt.imshow(confusion, cmap=plt.cm.Blues)
# ticks 坐标轴的坐标点
# label 坐标轴标签说明
indices = range(len(confusion))
# plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.rcParams['axes.unicode_minus'] = False
# 第一个是迭代对象，表示坐标的显示顺序，第二个参数是坐标轴显示列表
plt.xticks(indices, ['normal','Bot','Brute Force','DOS','PortScan','Web Attack'],size = 14)
plt.yticks(indices, ['normal','Bot','Brute Force','DOS','PortScan','Web Attack'],size = 14)

plt.colorbar(fraction=0.0455, pad=0.05)

# plt.rcParams两行是用于解决标签不能显示汉字的问题
#plt.rcParams['font.sans-serif']=['SimHei']
#plt.rcParams['axes.unicode_minus'] = False
plt.xlabel('True Label',fontsize = 16)
plt.ylabel('Predicted Label',fontsize = 16)
# plt.title('随机森林分类结果的混淆矩阵',fontsize = 16)

# 显示数据
for first_index in range(len(confusion)):    #第几行
    for second_index in range(len(confusion[first_index])):    #第几列
        plt.text(first_index, second_index, confusion[first_index][second_index],ha="center",color = 'y',size = 18)
# 在matlab里面可以对矩阵直接imagesc(confusion)
# 显示
plt.savefig("./DTMatrix.png",bbox_inches='tight')
plt.show()