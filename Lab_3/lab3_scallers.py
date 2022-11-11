import pandas as pd
# import matplotlib as plt
import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import load_breast_cancer

from sklearn.preprocessing import StandardScaler



from sklearn.neighbors import KNeighborsClassifier as kNN
from sklearn.svm import SVC

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix, accuracy_score,f1_score


from  sklearn.preprocessing import *
from sklearn.linear_model import *


from sklearn.tree import DecisionTreeClassifier as DT
from sklearn.tree import plot_tree



# docs

import numpy as np

import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib import cm

from sklearn.preprocessing import MinMaxScaler
# from sklearn.preprocessing import minmax_scale
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import PowerTransformer



X,y = load_breast_cancer(return_X_y=True, as_frame=True)

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state= 221, shuffle=False)

print(X_train.shape)
print(y_train.shape)

scalers = [StandardScaler(),MinMaxScaler(),MaxAbsScaler(),
           RobustScaler(),Normalizer(),QuantileTransformer(),
           PowerTransformer()
           ]

names = ['StandardScaler()','MinMaxScaler()','MaxAbsScaler()',
           'RobustScaler()','Normalizer()','QuantileTransformer()',
           'PowerTransformer'
           ]

results = []

for scaler in scalers:
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    temp = []
    models = [kNN(15), SVC(),DT()]
    for model in models:
        model.fit(X_train,y_train)
        y_pred = model.predict(X_test)
        # cm = confusion_matrix(y_test, y_pred)
        # print(cm)
        # print("f1 metric", f1_score(y_test,y_pred))
        temp.append(f1_score(y_test,y_pred))
    results.append(temp)

# print(results,sep='\n')

df = pd.DataFrame(results, columns = ['knn', 'SVC','DT'])
print(df)
print(max(df))
# scaler = StandardScaler()

# scaler.fit(X_train)
# X_train = scaler.transform(X_train)
# X_test = scaler.transform(X_test)