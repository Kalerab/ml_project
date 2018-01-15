import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
from matplotlib import pyplot as plt



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

#urcenie stupna polynomu
for polynom in range(2,4):
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

        poly = PolynomialFeatures(degree=polynom)
        X_ = poly.fit_transform(train_set)
        predict_ = poly.fit_transform(predict)

        clf = linear_model.LinearRegression()
        clf.fit(X_, train_out)
        pred_val = clf.predict(predict_)

        #ratanie trenovacej chyby - absolutna a percentualna chyba
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

    print("Average error for polynome of {}-degree is {} and percentage error is {}%".format(polynom,errtotal,pertotal))
    print()


#PLOTOVANIE


# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# axes = plt.gca()
# axes.set_xlim([xmin,xmax])
# axes.set_ylim([ymin,ymax])
#
#
# for i in range(3465):
#     if i%35==0:
#         sex_val = (train_set[i][0] * 0.85 + train_set[i][1] * 0.5 + train_set[i][2]) / 2
#         trainplot[i][0] = train_set[i][3] * train_set[i][5] * 100 + sex_val + train_set[i][4]
#         trainplot[i][1] = train_set[i][6] + (train_set[i][7] + train_set[i][8] + train_set[i][9]) * 0.3
#         trainplot[i][2] = train_out[i]
#         ax.scatter(trainplot[i][0], trainplot[i][1], trainplot[i][2], c='b', marker='o')
#
# for i in range(len(val)):
#     if i % 10==0:
#         sex_val = (predict[i][0] * 0.85 + predict[i][1] * 0.5 + predict[i][2]) / 2
#         predictplot[i][0] = predict[i][3] * predict[i][5] * 100 + sex_val + predict[i][4]
#         predictplot[i][1] = predict[i][6] + (predict[i][7] + predict[i][8] + predict[i][9]) * 0.3
#         predictplot[i][2] = val[i]
#         ax.scatter(predictplot[i][0], predictplot[i][1], predictplot[i][2], c='r', marker='o')
#
# pltX = 0
# pltY = 1
# pltZ = 2
#
# dim1 = 5
# dim2 = 3
#
# for i in range(len(val)):
#     if i % 10==0:
#         predictplot[i][0] = predict[i][dim1]
#         predictplot[i][1] = predict[i][dim2]
#         predictplot[i][2] = val[i]
#         ax.scatter(predictplot[i][pltX], predictplot[i][pltY], predictplot[i][pltZ], c='b', marker='o')
#
#
# for i in range(len(val)):
#     if i % 10==0:
#         predictplot[i][0] = predict[i][dim1]
#         predictplot[i][1] = predict[i][dim2]
#         predictplot[i][2] = pred_val[i]
#         ax.scatter(predictplot[i][pltX], predictplot[i][pltY], predictplot[i][pltZ], c='r', marker='o')
#
# if pltX == 0:
#     ax.set_xlabel(cols[dim1])
# elif pltX==1:
#     ax.set_xlabel(cols[dim2])
# else:
#     ax.set_xlabel('age')
#
# if pltY == 0:
#     ax.set_ylabel(cols[dim1])
# elif pltY == 1:
#     ax.set_ylabel(cols[dim2])
# else:
#     ax.set_ylabel('age')
#
# if pltZ == 0:
#     ax.set_zlabel(cols[dim1])
# elif pltZ == 1:
#     ax.set_zlabel(cols[dim2])
# else:
#     ax.set_zlabel('age')
#
# plt.show()