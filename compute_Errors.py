# Sample  MSE , MAE , RMSE , RAE , RSA , R2 Caculation 

from sklearn.metrics import r2_score
import numpy as np 

#Predicted and acyual values 
actualValue = [2, 4, 3, 2, 5, 6, 7]
predictedValue = [2, 3.5, 2.7, 2, 4.3, 6, 7.5]

# turn actual and prredicted values to numpy array 
actual = np.asanyarray(actualValue)
predicted = np.asanyarray(predictedValue)

#Compute MSE : Mean Square Error 
print("MSE : ",np.mean((actual-predicted)**2))

#Compute MAE : Mean Absolute Error 
print("MAE : ", np.mean(np.absolute(actual - predicted)))

#Compute RMSE : Root Mean Squre Error 
print("RMSE : ",np.square(np.mean((actual - predicted)**2)))

#Compute RAE : Relative Absolute Error 
    # for calculationg RAE we seprate operations :
        # 1. numerator : Sum of absolute value of Differentation of actual and predicted 
        # 2. denominator : Sum of absolute value of Differentation of actual and mean actual 
        # 3. RAE : numerator / denominator 
numerator = np.sum(np.absolute(actual - predicted))
denominator = np.sum(np.absolute(actual - np.mean(actual)))
print("RAE : " , numerator/denominator)
# print("RAE : %.2f" %(numerator/denominator))

#Compute RSE : Relative Square Error 
# for calculationg RAE we seprate operations :
        # 1. numerator1 : Sum of squared value of Differentation of actual and predicted 
        # 2. denominator1 : Sum of squared value of Differentation of actual and mean actual 
        # 3. RSE : numerator / denominator 
numerator1 = np.sum((actual - predicted)**2)
denominator1 = np.sum((actual - np.mean(actual))**2)
print("RSE : " , numerator1/denominator1)

#Compute R2 : R-Square 
    # R2 = 1 - RSE    ---  or ---- r2_score from sklearn 
print("R2 : " , r2_score(actual , predicted))

