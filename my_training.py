import pandas as pd
import numpy as np
import pickle

df=pd.read_csv("FuelConsumptionCo2.csv")

cdf=df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']]

msk=np.random.rand(len(df))<0.8
train=cdf[msk]
test=cdf[~msk]

from sklearn import linear_model
regr = linear_model.LinearRegression()
train_x = np.asanyarray(train[['ENGINESIZE']])
train_y = np.asanyarray(train[['CO2EMISSIONS']])
regr.fit (train_x, train_y)

test_x = np.asanyarray(test[['ENGINESIZE']])
test_y = np.asanyarray(test[['CO2EMISSIONS']])

#open a file where you want to store the data
file=open('model.pkl','wb')
pickle.dump(regr,file)
file.close()

