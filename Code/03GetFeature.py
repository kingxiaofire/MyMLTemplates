"""
1：Sample
2：Smote
3：Pretrain
4: Featureimportance
 """
import pandas as pd
from sklearn.model_selection import train_test_split 


df_0 = df[(df['target_type'] == 0)]
df_0 = df_0.sample(n=150000 ,replace=False ,weights=None ,random_state=None ,axis=0)
df_3 = df[(df['target_type'] == 3)]
df_3 = df_3.sample(n=150000 ,replace=False ,weights=None ,random_state=None ,axis=0)
df_5 = df[(df['target_type'] == 5)]
df_5 = df_5.sample(n=150000 ,replace=False ,weights=None ,random_state=None ,axis=0)

df_new = df_0.append(df_3).append(df_5).append(df[(df['target_type'] == 2)]).append(df[(df['target_type'] == 6)]).append(df[(df['target_type'] == 1)])

X = df_new.iloc[:,:-2]
Y = df_new.iloc[:,-1]

Xtrain,Xtest,Ytrain,Ytest = train_test_split(X,Y,test_size = 0.2,random_state = 7)

from imblearn.over_sampling import SMOTE
smote = SMOTE(n_jobs = -1 ,sampling_strategy = {1:15000,6:15000} ,random_state = 100) # Create 1500 samples for the minority class "4"
Xtrain, Ytrain = smote.fit_resample(Xtrain, Ytrain)
pd.Series(Ytrain).value_counts()


## Pretrain

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score,precision_recall_fscore_support
# Random Forest training and prediction
rf = RandomForestClassifier(random_state = 7,n_jobs = -1)
rf.fit(Xtrain,Ytrain) 

# # 求出预测和真实一样的数目
y_pred = gbm.predict(X_test)
true = np.sum(y_pred == y_test)
print('预测对的结果数目为：', true)
print('预测错的的结果数目为：', y_test.shape[0]-true)

rf_score=rf.score(Xtest,Ytest)
y_predict=rf.predict(Xtest)
y_true=Ytest
print('Accuracy of RF: '+ str(rf_score))
precision,recall,fscore,none= precision_recall_fscore_support(y_true, y_predict, average='weighted') 
print('Precision of RF: '+(str(precision)))
print('Recall of RF: '+(str(recall)))
print('F1-score of RF: '+(str(fscore)))
print(classification_report(y_true,y_predict,digits=6))
cm=confusion_matrix(y_true,y_predict)
f,ax=plt.subplots(figsize=(5,5))
sns.heatmap(cm,annot=True,linewidth=0.5,linecolor="red",fmt=".0f",ax=ax)
plt.xlabel("y_pred")
plt.ylabel("y_true")
plt.show()

threshold = 0.015
#Xselected = Xtrain[:,importances > threshold]
for i in range(Xtrain.shape[1]):
    if importances[indices[i]] > threshold:
        print("%2d %-*s %f" % (i + 1, 30, feat_labels[indices[i]], importances[indices[i]]))

from sklearn.feature_selection import SelectFromModel
selection= SelectFromModel(rf, threshold=threshold, prefit=True)
select_X_train = selection.transform(Xtrain)

selection_model =  RandomForestClassifier(n_estimators=100,random_state=3,n_jobs=-1)
selection_model.fit(select_X_train, Ytrain)

## GridSearchCV

from sklearn.model_selection import GridSearchCV
params = {
           #'n_estimators':range(400,1000,100)
           #,'criterion':('gini','entropy')
          # ,
    'max_depth':range(26,36,2)
         }
GS = GridSearchCV(
     selection_model
    ,params
    #,cv=5
    ,n_jobs=-1
)
GS.fit(select_X_train,Ytrain)
GS.best_params_, GS.best_score_

best_clf = RandomForestClassifier(n_estimators = 500
                                  ,random_state=3
                                  ,n_jobs=-1
                                  ,criterion='entropy'
                                  ,max_depth = 28
                                 )

rf = best_clf
rf.fit(Xtrain,Ytrain) 
rf_score=rf.score(Xtest,Ytest)
y_predict=rf.predict(Xtest)
y_true=Ytest
print('Accuracy of RF: '+ str(rf_score))
precision,recall,fscore,none= precision_recall_fscore_support(y_true, y_predict, average='weighted') 
print('Precision of RF: '+(str(precision)))
print('Recall of RF: '+(str(recall)))
print('F1-score of RF: '+(str(fscore)))
print(classification_report(y_true,y_predict))