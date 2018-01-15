import numpy as np
import pandas as pd
from sklearn import datasets, linear_model

#nacitanie dat a priradenie typu feature pohlavie
df = pd.read_csv("data.txt",dtype={"sex":"category"},delimiter=',')

#vytvorenie dummies pre pohlavie
df = pd.concat([df.drop('sex', axis=1), pd.get_dummies(df['sex'])], axis=1)

#shiftnutie 3 poslednych stlpcov na zaciatok
cols = df.columns.tolist()
cols = cols[-3:] + cols[:-3]
df = df[cols]
data = df.values
dfshuf = data

pertotal = 0
errtotal = 0
#pocet behov
runs = 75
for runnum in range(runs):

    #shuffle dat
    dfshuf = np.take(dfshuf, np.random.rand(dfshuf.shape[0]).argsort(), axis=0, out=dfshuf)


    train_set = dfshuf[0:3465, 0:10]
    train_out = dfshuf[0:3465, 10]
    predict = dfshuf[3465:4177, 0:10]
    val = dfshuf[3465:4177, 10]

    regr = linear_model.LinearRegression()
    regr.fit(train_set, train_out)
    pred_val = regr.predict(predict)

    error = 0
    per_error = 0
    for i in range(len(val)):
        error += abs(pred_val[i] - val[i])
        per_error += (abs(pred_val[i] - val[i])) / val[i]

    error = error / len(val)

    per_error = (per_error / len(val))

    errtotal += error
    pertotal += per_error

errtotal = errtotal / runs
pertotal = (pertotal / runs) * 100



print("Average error for linear regression is {} and percentage error is {}%".format(errtotal,pertotal))
print()