from scipy.io import wavfile
from scipy.fft import fft

from sklearn.preprocessing import LabelEncoder

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt # tutaj moze jeszcze krzyczec o pyplot!

import os
path = 'voices' # pomysl fajny gorzej ze ktos po biblioteke o tej samej nazwie nie siegnal
# dobrze by bylo do ostatecznej wersji jej uzyc
files = os.listdir(path)
fs = 16000
seconds = 3
X_raw = np.zeros((len(files), fs*seconds))
for i, file in enumerate(files):
    X_raw[i,:] = wavfile.read(f"{path}/{file}")[1] #i tutaj Windows suited path
    
#y = pd.read_excel('Genders_voises.xlsx').values # to tez dla nas posrednio jest bez kontekstu
y = pd.read_csv('res/Genders_voises.csv').values.ravel()

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)


X_fft = np.abs(fft(X_raw, axis=-1))/X_raw.shape[1]
low_cut = 50*seconds
hight_cut = 280*seconds
X_fft_cut = X_fft[:, low_cut:hight_cut]
fig, ax = plt.subplots(2,1)
ax[0].plot(np.arange(X_raw.shape[1]), X_raw[0,:])
ax[1].scatter(np.arange(X_raw.shape[1]),
X_fft[0,:], s = 0.5)
fig.tight_layout()

plt.show()

mean_num = 3
X_fft = np.reshape(
                    X_fft,
                    (
                    X_fft.shape[0],
                    X_fft.shape[1]//mean_num,
                    mean_num
                        )
                )
X_fft = X_fft.mean(axis=-1)
low_cut = 50
hight_cut = 280
X_fft_cut = X_fft[:, low_cut:hight_cut]
X_fft_cut = X_fft_cut/ np.expand_dims(X_fft_cut.max(axis=1),axis=-1)

# print(X_fft_cut.shape)
# print(X_fft_cut[0,:].shape)
print(X_fft_cut.shape)

plt.plot(np.arange(X_fft_cut.shape[1]), X_fft_cut[0,:]) # ten drugi argument jest nie halo tam bazowo  bylo0,:
# plt.plot(np.arange(X_fft_cut.shape[1]), X_fft_cut) 

#plt.plot(np.arange(X_fft_cut.shape[1]), X_fft_cut[0,:])
#  plt.plot(np.arange(X_fft_cut.shape[1],X_fft_cut[0,:]))

#jesli dobrze rozumiem to wybor dowolnego widma z dostepnych bedzie wlasciwa wizualizacja

#org
# plt.scatter(np.arange(X_raw.shape[1]),
# X_fft[0,:], s = 0.5)


#to zostawic
# plt.scatter(np.arange(X_fft_cut.shape[1]),
# X_fft_cut[0,:], s = 0.5)

plt.show()


#####################



# from sklearn.pipeline import Pipeline

# pipe =Pipeline([['transformer', PCA(62)],
#             ['scaler', StandardScaler()],
#             ['classifier', kNN(weights='distance')]]
#             )
# 3.pipe.fit(X_train, y_train)
# 4.y_pred = pipe.predict(X_test)
##
# Force import

from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier as kNN
from sklearn.tree import DecisionTreeClassifier as DT
from sklearn.svm import SVC


from sklearn.decomposition import PCA

from sklearn.metrics import confusion_matrix, accuracy_score,f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
##

classifiers = [kNN(), SVC(),DT()]

X_train, X_test,y_train, y_test = train_test_split(X_fft_cut, y,test_size=0.2, random_state=6, stratify=y)



pipe = Pipeline(steps = [
     ('transformer', PCA(0.95)),
     ('scaler', StandardScaler()),
    ('classifier', None) 
])

models_df = pd.DataFrame()


box_of_matrixes = []

# dla każdego typu modelu zmieniamy kolejne transformatory kolumn
for model in classifiers:
    temp = []
    for i in range(0,30):
        # odpowiednio zmieniamy jego paramety - dobieramy transformatory
        pipe_params = {
             'transformer' : PCA(0.95),
             'scaler': StandardScaler(),
            'classifier': model
        }
        pipe.set_params(**pipe_params)


        pipe.fit(X_train, y_train)   

        y_pred = pipe.predict(X_test)
        #print(confusion_matrix(y_test,y_pred).ravel())
        f1 = f1_score(y_test,y_pred)
        # print(f1)
        temp.append(confusion_matrix(y_test,y_pred).ravel())

    box_of_matrixes.append(temp)
        #tutaj moze byc inna bajka



# wlasciwie to nazwy delikatnie nie trafione prez wzglad na prace z np


kNNdf = np.stack(box_of_matrixes[0])

svm_df = np.stack(box_of_matrixes[1])
dt_df = np.stack(box_of_matrixes[2])

# print(f'dt_df {dt_df}')

print(f'kNN {kNNdf.mean(axis=0)}')

print(f'SVM {svm_df.mean(axis=0)}')

print(f'DT {dt_df.mean(axis=0)}')



# print(f'y = {y}')
# print(f'y = {y.shape[0]}')