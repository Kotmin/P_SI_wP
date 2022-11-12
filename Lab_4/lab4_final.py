import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.datasets import load_digits
from sklearn.decomposition import FastICA, PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.neighbors import KNeighborsClassifier as kNN
# from sklearn.metrics import confusion_matrix, accuracy_score

from sklearn.tree import DecisionTreeClassifier as DT
# from sklearn.tree import plot_tree
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.svm import SVC

from sklearn.metrics import confusion_matrix, accuracy_score,f1_score
#####

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time
from sklearn.datasets import load_digits
from sklearn.decomposition import FastICA, PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.neighbors import KNeighborsClassifier as kNN
# from sklearn.metrics import confusion_matrix, accuracy_score

from sklearn.tree import DecisionTreeClassifier as DT
# from sklearn.tree import plot_tree
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.svm import SVC

from sklearn.metrics import confusion_matrix, accuracy_score,f1_score
from sklearn.pipeline import Pipeline


#####



ionosphere=pd.read_csv('ionosphere_data.csv', header=None)
ionosphere.columns=['C'+str(i) for i in range(36)]

ionosphere.drop('C0', axis=1, inplace=True)
print(ionosphere.head())
print(ionosphere.shape)
print(ionosphere.iloc[:,-1].value_counts())




# scalers = [StandardScaler(),MinMaxScaler(),
#            RobustScaler(),
           
#            ]

X= ionosphere.iloc[:,:-1]
y= ionosphere.iloc[:,-1]


dim_red_alg = [PCA(),FastICA()]

models = [kNN(15), SVC(),DT(),RFC()]

X_train,X_test, y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=2022,stratify=y)

# cześć sluzaca do pozyskania odpowiedzi dot liczby probek

# pca_transform=PCA()
# pca_transform.fit(X_train)
# variances = pca_transform.explained_variance_ratio_
# cumulated_variances=variances.cumsum()
# plt.scatter(np.arange(variances.shape[0]), cumulated_variances)

# # plt.show()

# PC_num = (cumulated_variances<0.95).sum()+1 # wazne zeby dodac 1, poniewaz␣ chcemy zeby 95% bylo PRZEKROCZONE
# print("Aby wyjaśnić 95% wariancji, potrzeba "+str(PC_num)+' składowych głównych')

## Koniec

### I wracamy do przebiegu glownego tj porownania 


# pipe=Pipeline([
#     ['transformer', PCA(0.95)],
#     ['scaler', StandardScaler()],
#     ['classifier', kNN(weights='distance')]

# ])




# pipe.fit(X_train, y_train)
# y_pred=pipe.predict(X_test)
# # print(confusion_matrix(y_test, y_pred))
# # print(accuracy_score(y_test, y_pred))
# print(f1_score(y_test, y_pred))


#based on
# https://blog.prokulski.science/2020/10/10/pipeline-w-scikit-learn/
# Own ver

# scalers = [StandardScaler(),MinMaxScaler(),
#            RobustScaler(),
           
#            ]

scalers = Pipeline(steps = [
                            ('StandardScaler',StandardScaler()),
                            ('MinMaxScaler',MinMaxScaler())
])


classifiers = [kNN(15), SVC(),DT(),RFC()]
# classfiers = Pipeline(steps = [
#                             ('kNN',kNN()),
#                             ('SVC',SVC()),
#                             ('DT',DT()),
#                             ('RFC',RFC()),

# ])

transformers = [PCA(n_components=0.95),FastICA()]
models_df = pd.DataFrame()
 
# przygotowujemy pipeline
pipe = Pipeline(steps = [
    ('transform',None),
    ('scaler', scalers), # mniejszy pipeline
    ('classifier', None) # to ustalimy za moment
])
 
# dla każdego typu modelu zmieniamy kolejne transformatory kolumn
for model in classifiers:
    for num_tr in scalers:
        for cat_tr in transformers:
            # odpowiednio zmieniamy jego paramety - dobieramy transformatory
            pipe_params = {
                'transform': cat_tr,
                'scaler': num_tr,
                'classifier': model
            }
            pipe.set_params(**pipe_params)
 
            # trenujemy tak przygotowany model (cały pipeline) mierząc ile to trwa
            start_time = time.time()
            pipe.fit(X_train, y_train)   
            end_time = time.time()
 
            # sprawdzamy jak wyszło
            score = pipe.score(X_test, y_test)
            y_pred = pipe.predict(X_test)
            #tutaj moze byc inna bajka
            f1 = f1_score(y_test,y_pred)
 
            # zbieramy w dict parametry dla Pipeline i wyniki
            param_dict = {
                        'classifier': model.__class__.__name__,
                        'scaler_name': num_tr.__class__.__name__,
                        'transform': cat_tr.__class__.__name__,
                        'score': score,
                        'time_elapsed': end_time - start_time,
                        'f1 score': f1,
            }
 
            models_df = models_df.append(pd.DataFrame(param_dict, index=[0]))
 
models_df.reset_index(drop=True, inplace=True)

#print(models_df)
print(models_df.sort_values(['score'],ascending=False))