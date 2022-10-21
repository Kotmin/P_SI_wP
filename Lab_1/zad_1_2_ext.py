
import pandas as pd
import numpy as np

data = pd.read_csv("practice_lab_1.csv",sep=';')
data.to_csv("practice_lab_1_1.csv",index=False)
df = pd.read_csv("practice_lab_1_1.csv")

df.columns = df.columns.str.replace(' ', '_')



label_array = df.keys().values

array_of_values = df.to_numpy()

# print(label_array)
print(array_of_values)
#ind_df = df['col_a'].apply(lambda x: x*2) - df[col_b']



#print(df.head())

# Podpunkt 1

print("podpunt 1")

#par
tablica_par =np.array(df[df.columns[::]][::2].values)
#n par
tablica_npar = np.array(df[df.columns[::]][1::2].values)

print(tablica_npar)





print("\nPar: \n")
print(tablica_par)

wynikowa = np.subtract(tablica_par,tablica_npar)

print(f"\nWynikowa:\n{wynikowa}")

#P3 antoher

print(df.var())

#p6 another version
print("\nIle jest el.wiekszych od sredniej\n")
print(df.agg(lambda x : (x> x.mean()).sum()))
#Tu się az prosi o zahaczenie o sql'a

# 7
print("\nNajwieksze elementy sa w kolumnach\n")
print(pd.unique(df.loc[df.index].idxmax(axis=1)))
#obliczeniowo sredrnie

print("\nNajwięcej zer\n")
print(pd.unique(df.loc[df.index].eq(0,axis=0).idxmax(axis=1)))


# df[df.columns[::]][::2].values.sum(axis=1)
# df[df.columns[::]][1::2].values.sum(axis=1)

#

print("\nTam gdzie suma ehh podpunkt 9ty po prostu\n")
print(pd.unique(df.loc[df.index].eq(0,axis=0).idxmax(axis=1)))