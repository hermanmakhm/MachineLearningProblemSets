import pandas as pd

#read dataset
df = pd.read_csv('KNN_data.csv', delimiter = ',', names = ['x_1', 'x_2', 'Result'])
a = float(input('What is your x_1 point? '))
b = float(input('What is your x_2 point? '))
k = int(input('What is your k value? '))

#find distance
df.loc[:,'dist'] = ((a-df.loc[:,'x_1'])**2)+((b-df.loc[:,'x_2'])**2)

#new dataframe with lowest (k*2)-1 distances
df1 = df.nsmallest((k*2)-1,'dist')

#classify point by finding majority
if len(df1[df1['Result'] == 0]) > len(df1[df1['Result'] == 1]):
    print('Your point is classified as 0.')
else:
    print('Your point is classified as 1.')