import pandas as pd
import math
import numpy as np

#read dataset
df = pd.read_csv('titanic_medium.csv', delimiter = ',', index_col = 0)

#change sex and class into 2 and 3 columns respectively where 1 = True, 0 = False
df = pd.concat([pd.get_dummies(df['Sex']), pd.get_dummies(df['Pclass']), df['Survived']], axis=1)

#delete male column since it is equal and opposite of female column
df = df.drop(['male'], axis=1)

#change column heads for easy looping
df.columns = [0, 1, 2, 3, 4]

print('We let columns 0, 1, 2, 3 be Female, Class1, Class2, Class3, respectively, and label them with 1 = True and 0 = False.')

#initial entropy can be calculated as follows
#this is uneeded, as minimum avg. entropy will always be the maximum info gained
#a = df.groupby(4)[4].count().tolist()
#ient = -((a[0]/len(df))*math.log(a[0]/len(df),2)+(a[1]/len(df))*math.log(a[1]/len(df),2))

#null array creation
B = np.zeros((4,4),dtype = np.int16)
FinEnt = np.zeros((2,4))
AvgEnt = np.zeros(4)

#first "decision"
for i in range (0,4):
    #given classification, find number of survived vs. dead
    for j in range (0, len(df)):
        if df.loc[j,i] == 0:
            if df.loc[j,4] == 1:
                B[0,i] += 1
            elif df.loc[j,4] == 0:
                B[1,i] += 1
        elif df.loc[j,i] == 1:
            if df.loc[j,4] == 1:
                B[2,i] += 1
            elif df.loc[j,4] == 0:
                B[3,i] += 1
    #final entropy of each classification
    FinEnt[0,i] = (-B[0,i]*math.log(B[0,i]/(B[0,i]+B[1,i]),2)-B[1,i]*math.log(B[1,i]/(B[0,i]+B[1,i]),2))/(B[0,i]+B[1,i])
    FinEnt[1,i] = (-B[2,i]*math.log(B[2,i]/(B[2,i]+B[3,i]),2)-B[3,i]*math.log(B[3,i]/(B[2,i]+B[3,i]),2))/(B[2,i]+B[3,i])
    #average entropy
    AvgEnt[i] = (FinEnt[0,i]+FinEnt[1,i])/2

#print first "decision" based on {highest info gain}≡{lowest average entropy}
print('We achieve the most information gain by first splitting column', AvgEnt.argmin())

#second "decision"; begin by setting two new dataframes that separate based on first "decision" and reset dataframe indicies
df1 = df.loc[df[AvgEnt.argmin()] == 0].reset_index(drop=True)
df2 = df.loc[df[AvgEnt.argmin()] == 1].reset_index(drop=True)

#create null arrays
B1 = np.zeros((4,4),dtype = np.int16)
FinEnt1 = np.zeros((2,4))
AvgEnt1 = np.zeros(4)
B2 = np.zeros((4,4),dtype = np.int16)
FinEnt2 = np.zeros((2,4))
AvgEnt2 = np.zeros(4) 

for i in range (0,4):
    #only do calculation if column was not the previous "decision" column
    if i !=AvgEnt.argmin():
        #given classification, find number of survived vs. dead
        for j in range (0, len(df1)):
            if df1.loc[j,i] == 0:
                if df1.loc[j,4] == 1:
                    B1[0,i] += 1
                elif df1.loc[j,4] == 0:
                    B1[1,i] += 1
            elif df1.loc[j,i] == 1:
                if df1.loc[j,4] == 1:
                    B1[2,i] += 1
                elif df1.loc[j,4] == 0:
                    B1[3,i] += 1
        #final entropy of each classification
        FinEnt1[0,i] = (-B1[0,i]*math.log(B1[0,i]/(B1[0,i]+B1[1,i]),2)-B1[1,i]*math.log(B1[1,i]/(B1[0,i]+B1[1,i]),2))/(B1[0,i]+B1[1,i])
        FinEnt1[1,i] = (-B1[2,i]*math.log(B1[2,i]/(B1[2,i]+B1[3,i]),2)-B1[3,i]*math.log(B1[3,i]/(B1[2,i]+B1[3,i]),2))/(B1[2,i]+B1[3,i])
        #average entropy
        AvgEnt1[i] = (FinEnt1[0,i]+FinEnt1[1,i])/2
    else:
        AvgEnt1[i] = FinEnt[0,i]
    
for i in range (0,4):
    #only do calculation if column was not the previous "decision" column
    if i !=AvgEnt.argmin():
        #given classification, find number of survived vs. dead
        for j in range (0, len(df2)):
            if df2.loc[j,i] == 0:
                if df2.loc[j,4] == 1:
                    B2[0,i] += 1
                elif df2.loc[j,4] == 0:
                    B2[1,i] += 1
            elif df2.loc[j,i] == 1:
                if df2.loc[j,4] == 1:
                    B2[2,i] += 1
                elif df2.loc[j,4] == 0:
                    B2[3,i] += 1
        #final entropy of each classification
        FinEnt2[0,i] = (-B2[0,i]*math.log(B2[0,i]/(B2[0,i]+B2[2,i]),2)-B2[1,i]*math.log(B2[1,i]/(B2[0,i]+B2[1,i]),2))/(B2[0,i]+B2[1,i])
        FinEnt2[1,i] = (-B2[2,i]*math.log(B2[2,i]/(B2[2,i]+B2[3,i]),2)-B2[3,i]*math.log(B2[3,i]/(B2[2,i]+B2[3,i]),2))/(B2[2,i]+B2[3,i])
        #average entropy
        AvgEnt2[i] = (FinEnt2[0,i]+FinEnt2[1,i])/2
    else:
        AvgEnt2[i] = FinEnt[1,i]

#print second "decisions'
print('Where column', AvgEnt.argmin(),'has the value 0, we then gain the most information by splitting column', AvgEnt1.argmin())
print('Where column', AvgEnt.argmin(),'has the value 1, we then gain the most information by splitting column', AvgEnt2.argmin())
