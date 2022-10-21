 
import pandas as pd
import numpy as np

data = pd.read_csv("practice_lab_1.csv",sep=';')
data.to_csv("practice_lab_1_1.csv",index=False)
df = pd.read_csv("practice_lab_1_1.csv")

df.columns = df.columns.str.replace(' ', '_')



label_array = df.keys().values

print(label_array)
#ind_df = df['col_a'].apply(lambda x: x*2) - df[col_b']



print(df.head())

#df['kolumna_1'] = df['kolumna_1'] * 2


#Wyrzucanie kolumny na ekran
#print(df.kolumna_1.to_string(index=False))
#Podpunkt 1

#EVEN
print(df[df.columns[::2]][::2])
#ODD
print(df[df.columns[1::2]][1::2])


print("BreakPoint")

# Podpunkt 2

#https://stackoverflow.com/questions/71778472/calculate-every-row-in-a-pandas-dataframe-with-a-value-specific-to-that-row


srednia = df.mean()
odchylenie_std = df.std()

df.update(df.filter(like='col').sub(srednia).div(odchylenie_std))

print("Podpunkt 2 wynik:\n")
print(df.head())

# Tu coś zdaje się jeszcze nie banglac

#Podpunkt 3
print("\n")
print(df['kolumna_1'].mean())


#With normalization
#df.std(ddof=0)
print(df.std(axis=0)) # 0 po kolumnach 1 po wierszach
#print(df['kolumna_1'].std())

print("\nPodpunkt 3 wynik:\n")


print(df.describe())

# Podpunkt 4

# df['pct_change']=(df.pct_change())
# df['change']=df.diff()

print("\n Zmiana: \n")
print(df.diff())

# Podpunkt 5

print(df.diff().mean().max(axis=0))

#data.style_format

#magic = data.mul(2).to_frame('czary')

#print(magic.head())

print(df.head())