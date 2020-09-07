# -*- coding: utf-8 -*-
"""genre (1).ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1c5s4BSk_fPCVpNicolZHNiZigOENYHEl
"""

# import html2text
import os
import pandas as pd
import numpy as np
import re
from sklearn import preprocessing
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import plot_confusion_matrix
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

df=pd.read_csv("/content/drive/My Drive/Colab Notebooks/feature_lables.csv")

from google.colab import drive
drive.mount('/content/drive')

df.head()

sent_1=df["Senti_S1"].to_list()
sent_2=df["Senti_S2"].to_list()
sent_3=df["Senti_S3"].to_list()
sent_4=df["Senti_E1"].to_list()
sent_5=df["Senti_E2"].to_list()
sent_6=df["Senti_E3"].to_list()
avg_S_len=df["Avg_S_len"].to_list()
flesch=df["Flesch"].to_list()
w_count=df["W_count"].to_list()
noun_Cnt=df["Noun_Cnt"].to_list()
s_count=df["S_count"].to_list()

feat=list(zip(sent_1,sent_2,sent_3,sent_4,sent_5,sent_6,avg_S_len,flesch,w_count,noun_Cnt,s_count))

labels=df['guten_genre'].to_list()

set_labels=list(set(labels))

Counter(labels)

feat=[list(_) for _ in feat]

encoder = preprocessing.LabelEncoder()
transformed_label=encoder.fit_transform(labels) #LabelEncoder()

label_i=(encoder.inverse_transform(transformed_label))
q=list(zip(transformed_label,label_i))

mapping={}
for a,b in set(q):
    mapping.update({a:b})

mapping

mapping_order=['Allegories','Christmas Stories','Detective and Mystery','Ghost and Horror','Humorous and Wit and Satire','Literary','Love and Romance','Sea and Adventure','Western Stories']
len(mapping_order)

X_train, X_test, Y_train, Y_test = train_test_split(feat, transformed_label, stratify=transformed_label,test_size=0.25)

model=svm.SVC(kernel='linear')
model.fit(X_train,Y_train)

preds=model.predict(X_test)

print('accuracy is: ',accuracy_score(Y_test, preds))

print(classification_report(Y_test,preds))

print(confusion_matrix(Y_test,preds))

import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

df_cm = pd.DataFrame(confusion_matrix(Y_test,preds), mapping_order, mapping_order)
sn.set(font_scale=1.4) # for label size
sn.heatmap(df_cm, annot=True, annot_kws={"size": 14},fmt='d') # font size
# plt.xticks(['Detective and Mystery', 'Literary', 'Western Stories',
#        'Ghost and Horror', 'Sea and Adventure', 'Christmas Stories',
#        'Love and Romance', 'Allegories', 'Humorous and Wit and Satire'],rotation=90)
plt.show()

from sklearn.naive_bayes import GaussianNB 

  
classifier = GaussianNB(); 
classifier.fit(X_train, Y_train)

preds=classifier.predict(X_test)

print('accuracy is: ',accuracy_score(Y_test, preds))

print(classification_report(Y_test,preds))

df_cm = pd.DataFrame(confusion_matrix(Y_test,preds), mapping_order, mapping_order)
sn.set(font_scale=1.4) # for label size
sn.heatmap(df_cm, annot=True, annot_kws={"size": 14} ,fmt='d') # font size
# plt.xticks(['Detective and Mystery', 'Literary', 'Western Stories',
#        'Ghost and Horror', 'Sea and Adventure', 'Christmas Stories',
#        'Love and Romance', 'Allegories', 'Humorous and Wit and Satire'],rotation=90)
plt.show()

X_train, X_test, Y_train, Y_test = train_test_split(feat, transformed_label, stratify=transformed_label,test_size=0.70)

from sklearn.ensemble import GradientBoostingClassifier

lr_list = [0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1]

for learning_rate in lr_list:
    gb_clf = GradientBoostingClassifier(n_estimators=20, learning_rate=learning_rate, max_features=2, max_depth=2, random_state=0)
    gb_clf.fit(X_train, Y_train)
    print("Learning rate: ", learning_rate)
    print("Accuracy score (training): {0:.3f}".format(gb_clf.score(X_train, Y_train)))
    print("Accuracy score (validation): {0:.3f}".format(gb_clf.score(X_test, Y_test)))
    preds=gb_clf.predict(X_test)
    df_cm = pd.DataFrame(confusion_matrix(Y_test,preds), mapping_order, mapping_order)
    sn.set(font_scale=1.4) # for label size
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 14}) # font size
    # plt.xticks(['Detective and Mystery', 'Literary', 'Western Stories',
    #        'Ghost and Horror', 'Sea and Adventure', 'Christmas Stories',
    #        'Love and Romance', 'Allegories', 'Humorous and Wit and Satire'],rotation=90)
    plt.pause(20)
    plt.show()
    plt.close()

from xgboost import XGBClassifier

xgb_clf = XGBClassifier()
xgb_clf.fit(np.array(X_train), Y_train)

score = xgb_clf.score(np.array(X_test), Y_test)
print(score)

model_rbf=svm.SVC(kernel='rbf')
model_rbf.fit(X_train,Y_train)

preds=model_rbf.predict(X_test)

print('accuracy is: ',accuracy_score(Y_test, preds))

X_train, X_test, Y_train, Y_test = train_test_split(feat, transformed_label, stratify=transformed_label,test_size=0.70)

from collections import Counter

Counter(list(transformed_label))

z=feat.copy()

to_pop=0
l=list(transformed_label).copy()
new_l=[]
new_z=[]
for i in range(len(l)):
    if l[i]!=5 and to_pop<950:
        new_l.append(l[i])
        new_z.append(feat[i])
    if to_pop>=950:
        new_l.append(l[i])
        new_z.append(feat[i])
        #z.pop(i)
        #l.pop(i)
    to_pop+=1

len(new_l)

len(new_z)

Counter(new_l)

to_pop=0
l=list(new_l).copy()
new_l2=[]
new_z2=[]
for i in range(len(l)):
    if l[i]!=2 and to_pop<200:
        new_l2.append(l[i])
        new_z2.append(new_z[i])
    if to_pop>=200:
        new_l2.append(l[i])
        new_z2.append(new_z[i])
        #z.pop(i)
        #l.pop(i)
    to_pop+=1

len(new_z2)

Counter(new_l2)

X_train, X_test, Y_train, Y_test = train_test_split(new_z2, new_l2, test_size=0.20)

model_lin=svm.SVC(kernel='linear')
model_lin.fit(X_train,Y_train)

preds=model_lin.predict(X_test)

print('accuracy is: ',accuracy_score(Y_test, preds))



