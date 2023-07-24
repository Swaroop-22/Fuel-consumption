# -*- coding: utf-8 -*-
"""
Fuel Consumption using multiple linear regression
Created on Mon Jan 25 15:58:00 2021

@author: Swaroop Honrao
"""
#import library
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl

%matplotlib inline  

#import dataset
df = pd.read_csv('FuelConsumptionCo2.csv')
df.head()  #to check top values
cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_CITY','FUELCONSUMPTION_COMB','FUELCONSUMPTION_HWY','CO2EMISSIONS']]
cdf.head(9)
x = cdf.iloc[:, :-1].values
y = cdf.iloc[:, -1].values

#splitting dataset into training set and testing set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 0)

#Linear regressor
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)
print('coefficients: ', regressor.coef_)

#predicted results
y_pred = regressor.predict(x_test)
print('residual error: %2f' %np.mean(y_pred-y_test)**2)
print('variance score: %2f' %regressor.score(x_test,y_test)) # 1 is perfecr prediction

import statsmodels.api as sm
x = np.append(arr = np.ones((1067,1)).astype(int), values=x, axis = 1)

x_opt = x[:, [0,1,2,3,4,5]]
regressor_OLS = sm.OLS(endog = y , exog = x_opt).fit()
regressor_OLS.summary()


x_opt = x[:, [0,1,4,5]]
regressor_OLS = sm.OLS(endog = y , exog = x_opt).fit()
regressor_OLS.summary()


