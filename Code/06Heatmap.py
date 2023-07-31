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