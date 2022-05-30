import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd
"""
    PyLab is a procedural interface to the Matplotlib object-oriented plotting library.
    Matplotlib is the whole package; matplotlib.pyplot is a module in Matplotlib;
    and PyLab is a module that gets installed alongside Matplotlib.

    PyLab is a convenience module that bulk imports matplotlib.pyplot
    (for plotting) and NumPy (for Mathematics and working with arrays)
    in a single name space. Although many examples use PyLab, it is no longer recommended.
""" 
import pylab as pl 
from sklearn.preprocessing import PolynomialFeatures 
from sklearn import linear_model
from sklearn.metrics import r2_score

df = pd.read_csv("/home/asma/Documents/Programing/AI/ML_with_Jadi/Codes/Machine-Learning-with-python/Regression/FuelConsumption.csv")
print(df.head())

cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
print(cdf.head())

plt.scatter(cdf.ENGINESIZE , cdf.CO2EMISSIONS , color= 'blue')
plt.xlabel("ENGINESIZE")
plt.ylabel("CO2 EMISSONS")
# plt.show()


# Create Test and Train Data 

mask = np.random.rand(len(cdf)) < 0.8 
train = cdf[mask]
test = cdf[~mask]


# Polynomial Regression 

train_x = np.asanyarray(train[['ENGINESIZE']])
train_y = np.asanyarray(train[['CO2EMISSIONS']])

test_x = np.asanyarray(test[['ENGINESIZE']])
test_y = np.asanyarray(test[['CO2EMISSIONS']])


poly = PolynomialFeatures(degree=2)
train_x_poly = poly.fit_transform(train_x)



clf = linear_model.LinearRegression()
train_y_ = clf.fit(train_x_poly, train_y)
# The coefficients
print ('Coefficients: ', clf.coef_)
print ('Intercept: ',clf.intercept_)


plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS,  color='blue')
XX = np.arange(0.0 ,10.0 ,0.1)
YY = clf.intercept_[0] + clf.coef_[0][1] *XX + clf.coef_[0][2] * np.power(XX , 2)
plt.plot(XX , YY , '-r')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()

# Evaluation 

test_x_poly = poly.fit_transform(test_x)
test_y_= clf.predict(test_x_poly)

print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_ - test_y)))
print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_ - test_y) ** 2))
print("R2-score: %.2f" % r2_score(test_y,test_y_ ) )
