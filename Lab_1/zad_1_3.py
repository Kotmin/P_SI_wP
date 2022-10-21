import numpy as np 
from matplotlib import pyplot as plt 






plt.title("Zadanko 1.3") 
plt.xlabel("x") 
plt.ylabel(f"f(x)=") 
#x = np.arange(1,11) 

x = np.arange(-5,5,0.01)

# Pierwszy 

y = np.tanh(x)

plt.ylabel(f"f(x)=tanh(x)") 

plt.plot(x,y) 
plt.show()

# Drugi

plt.ylabel(f"np.exp(x) - np.exp(-x)) / ((np.exp(x) + np.exp(-x))")

y = (np.exp(x) - np.exp(-x)) / ((np.exp(x) + np.exp(-x)))

plt.plot(x,y) 
plt.show()


# # Trzeci
plt.ylabel(r"np.where(x>0,x,0)")
#y = np.concatenate((x[x>0]),(0[x<=0]))
y = np.where(x>0,x,0)
plt.plot(x,y) 
plt.show()

# # Czwarty
plt.ylabel(r"np.where(x>0,x,(np.exp(x)-1))")
y = np.where(x>0,x,(np.exp(x)-1))
plt.plot(x,y) 
plt.show()


