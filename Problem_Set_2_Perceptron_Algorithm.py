import pandas as pd
import matplotlib.pyplot as plt
import random
import numpy as np

#read dataset
df = pd.read_csv('perceptron_data.csv', delimiter = ',', names = ['x_1', 'x_2', 'Result'])
#tiny step eta
eta = 0.001

#pick random a,b between (-1,0) and random c between (0,1)
a = random.uniform(-1,0)
b = random.uniform(-1,0)
c = random.uniform(0,1)

#iterate update perceptron algorithm 1,000,000 times when there is an error
for i in range (0,1000000):
    row = random.randint(0,len(df)-1)
    #find if there is error, if so, execute update algorithm
    if df.loc[row,'Result'] == 1 and a*df.loc[row,'x_1']+b*df.loc[row,'x_2']+c < 0:
        a = a + (eta * df.loc[row,'x_1'])
        b = b + (eta * df.loc[row,'x_2'])
        c = c + eta
    elif df.loc[row,'Result'] == 0 and a*df.loc[row,'x_1']+b*df.loc[row,'x_2']+c > 0:
        a = a - (eta * df.loc[row,'x_1'])
        b = b - (eta * df.loc[row,'x_2'])
        c = c - eta

#print weights
print(a,b,c)

#plot line
x = np.linspace(0,1)
plt.plot(x,(-a*x-c)/b)

#plot points
plt.scatter(df.x_1, df.x_2, c=df.Result)
plt.show()
