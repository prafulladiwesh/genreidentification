
import pandas as pd
import pprint as pp
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn import metrics
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import matplotlib.pyplot as plt
from matplotlib import figure
import seaborn as sns;sns.set()
from sklearn.metrics import confusion_matrix
import os

df = pd.read_csv('feature_lables.csv', encoding='latin1', sep=',')

thisdict = {'Detective and Mystery': 1, 'Literary': 2, 'Western Stories': 3, 'Ghost and Horror': 4,
            'Christmas Stories': 5, 'Love and Romance': 6, 'Sea and Adventure': 7, 'Allegories': 8,
            'Humorous and Wit and Satire': 9}

labels=['Detective and Mystery', 'Literary', 'Western Stories', 'Ghost and Horror',
            'Christmas Stories', 'Love and Romance', 'Sea and Adventure', 'Allegories',
            'Humorous and Wit and Satire']

df1=df.to_numpy()
BOOKS_PATH = "books"

n = len(df1)
f = len(df1[0])

Y=[]
X=[]
for i,row in enumerate(df1):

    X1=df1[i,1:f-1]
    X.append(X1)
    Y1=df1[i,-1]
    label_id = thisdict.get(Y1)
    Y.append(label_id)

##Features & Labels
X=np.array(X)
Y=np.array(Y)


x_train,x_test,y_train,y_test= train_test_split(X,Y, train_size=.70,test_size=.30)

split_no=len(x_train)
kf=KFold(n_splits=split_no, random_state=None, shuffle=True)

accuracy_nb=0
scores_nb=0
error_nb=0
accuracy_svm=0
scores_svm=0
error_svm=0
for train_index, test_index in kf.split(x_train,y_train):


    X_train1, X_val = x_train[train_index], x_train[test_index]
    Y_train1, Y_val = y_train[train_index], y_train[test_index]
    #Guassian NB
    clf = GaussianNB()
    clf.fit(X_train1, Y_train1)
    pred_nb=clf.predict(X_val)
    acc_nb=metrics.accuracy_score(Y_val,pred_nb)
    accuracy_nb = accuracy_nb + acc_nb
    scores_nb=1-acc_nb
    error_nb = error_nb + scores_nb


    # Linear SVM
    clf_svm = svm.SVC(kernel='linear')
    clf_svm.fit(X_train1, Y_train1)
    pred_svm = clf_svm.predict(X_val)
    acc_svm = metrics.accuracy_score(Y_val, pred_svm)
    accuracy_svm = accuracy_svm + acc_svm
    scores_svm = 1 - acc_svm
    error_svm = error_svm + scores_svm

print("NB Model Error  Count:",error_nb)
print("NB Model Accuracy Count:",accuracy_nb)
print("SVM Model Error  Count:",error_svm)
print("SVM Model Accuracy Count:",accuracy_svm)


if (accuracy_svm<accuracy_nb):

    # clf_test.fit(x_train,y_train)
    pred = clf.predict(x_test)
    test_acc = metrics.accuracy_score(y_test, pred)
    print("Test_Acc:", test_acc)
    F1_Score = f1_score(y_test, pred, average='weighted')
    print("Weighted F1 Score:", F1_Score)
else:
    # clf_test.fit(x_train,y_train)
    pred = clf_svm.predict(x_test)
    test_acc = metrics.accuracy_score(y_test, pred)
    print("Test_Acc:", test_acc)
    F1_Score = f1_score(y_test, pred, average='weighted')
    print("Weighted F1 Score:", F1_Score)


mat=confusion_matrix(y_test, pred)
ax=sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False,xticklabels=labels,yticklabels=labels,linewidths=.5)
plt.xlabel('True')
plt.ylabel('Predicted')
plt.show()