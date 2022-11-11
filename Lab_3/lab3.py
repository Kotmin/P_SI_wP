# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib as plt

from sklearn.datasets import load_breast_cancer

from sklearn.preprocessing import StandardScaler



from sklearn.neighbors import KNeighborsClassifier as kNN
from sklearn.svm import SVC

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix, accuracy_score,f1_score


# 3.4 

 # [ [TP,FP],
 #   [FP,TN],
 # ]

tab1_kNN =  [[7,26],
              [17,73],
             ]
tab2_SVM =  [[0,33],
              [0,90],
             ]             

def calc_sensivity(prepared_tab):
    return ((prepared_tab[0][0])/(prepared_tab[0][0]+prepared_tab[0][1]))

def calc_precision(prepared_tab):
    return (prepared_tab[0][0])/(prepared_tab[0][0]+prepared_tab[1][0]) if (prepared_tab[0][0]+prepared_tab[1][0]) else 0.0 

#Literowka w skrypcie byla
def calc_specificity(prepared_tab):
    return ((prepared_tab[1][1])/(prepared_tab[1][0]+prepared_tab[1][1]))


def calc_accuracy(prepared_tab):
    return ((prepared_tab[0][0] + prepared_tab[1][1])/(prepared_tab[0][0]+prepared_tab[0][1] + prepared_tab[1][0]+prepared_tab[1][1]))


def calc_confusion_matrix(prepared_tab):
    return [[calc_sensivity(prepared_tab),calc_precision(prepared_tab)],
            [calc_sensivity(prepared_tab),calc_accuracy(prepared_tab)]]



print(calc_sensivity(tab1_kNN))
print(calc_confusion_matrix(tab1_kNN))
print(calc_confusion_matrix(tab2_SVM))


#

data = load_breast_cancer()
# X,y = load_breast_cancer(returnX_y=True, as_frame=True)
#zwraca nam serie z nazwami 

dane = pd.DataFrame(data.data, columns=data.feature_names)

dane['target']=data.target


X, y = dane.iloc[:,:-1], dane.iloc[:, -1]


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state= 221, shuffle=False)

scaler = StandardScaler()

scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)




models = [kNN(), SVC()]
for model in models:
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    print("accuracy score",accuracy_score(y_test,y_pred),"f1 metric", f1_score(y_test,y_pred))


# ctrl + i go to docs


from sklearn.tree import DecisionTreeClassifier as DT
from sklearn.tree import plot_tree


model = DT(max_depth=5)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
cm2 = confusion_matrix(y_test, y_pred)
print(cm2)
print("accuracy score",accuracy_score(y_test,y_pred),"f1 metric", f1_score(y_test,y_pred))
from matplotlib import pyplot as plt
plt.figure(figsize=(20,10))
tree_vis = plot_tree(model,feature_names=
dane.columns[:-1],class_names=['N', 'Y'], fontsize = 10)

plt.show()


# Tests for diff ,,somsiad" number



somsiads = range(1,30)

f1_metrics = []
x = []
for somsiad in somsiads:
    model = kNN(somsiad)
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    #cm = confusion_matrix(y_test, y_pred)
    #print(cm)
    f1_metrics.append(f1_score(y_test,y_pred))
    x.append(somsiad)


print(f1_metrics)

print(max(f1_metrics))

#DT

tree_depth = range(1,20)

f1_metrics = []

for tree in tree_depth:
    model = DT(max_depth=tree)
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    f1_metrics.append(f1_score(y_test,y_pred))


print(f'For DT{f1_metrics}')

print(max(f1_metrics))

#SVC

kernels = ['linear', 'poly', 'rbf', 'sigmoid']

f1_metrics = []

for k in kernels:
    model = SVC(kernel=k)
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    f1_metrics.append(f1_score(y_test,y_pred))
    

#rbw was def

print(f'For SVC {f1_metrics}')
print(max(f1_metrics))