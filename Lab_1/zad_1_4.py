import pandas as pd
import numpy as np

from matplotlib import pyplot as plt 


data = pd.read_csv("practice_lab_1.csv",sep=';')
data.to_csv("practice_lab_1_1.csv",index=False)
df = pd.read_csv("practice_lab_1_1.csv")

df.columns = df.columns.str.replace(' ', '_')




#ind_df = df['col_a'].apply(lambda x: x*2) - df[col_b']

# ładniej wytłumaczone
# https://www.sfu.ca/~mjbrydon/tutorials/BAinPy/08_correlation.html


# print(df.head())

corr = df.corr()

print(corr)

# print(corr['kolumna_1']['kolumna_1'])


# rozpocznijmy od przedstawienia pierwszej z drugim
x = df['kolumna_1']
y = df['kolumna_2']

plt.scatter(x,y)
plt.plot(np.unique(x),np.poly1d(np.polyfit(x,y,1))
        (np.unique(x)),color = 'red')

plt.show()
# fig, ax = plt.subplots(7,7, figsize = (10, 10))


# plt.rcParams.update({'figure.figsize':(10,8), 'figure.dpi':100})

# fig,ax = plt.subplots(7,7, figsize = (10, 10))

counter = 1
for a in range(1,8):
    for b in range(1,8):
        # if x==y:
        #         continue
        x = df[f'kolumna_{a}']
        y = df[f'kolumna_{b}']

        plt.subplot(7,7,counter)
        plt.scatter(x,y)
        plt.plot(np.unique(x),np.poly1d(np.polyfit(x,y,1))
                (np.unique(x)),color = 'red')
        counter+=1


# plt.plot(ax)
plt.show()

# Dobra nie czajen może z rana

# ax[0,0].scatter(x,y)
# ax[0,1].plot(x,y)
# ax[1,0].hist(y)
# ax[1,1].boxplot(y)