import quandl
import pandas as pd
import numpy as np
import datetime

from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
import pickle


df = quandl.get("WIKI/AMZN")



df = df[['Adj. Close']]


forecast_out = int(30) # predicting 30 days into future
df['Prediction'] = df[['Adj. Close']].shift(-forecast_out) #  label column with data shifted 30 units up


X = np.array(df.drop(['Prediction'], 1))
X = preprocessing.scale(X)



X_forecast = X[-forecast_out:]
X = X[:-forecast_out]



y = np.array(df['Prediction'])
y = y[:-forecast_out]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)


# Create and train the Support Vector Machine (Regressor) 
svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1) 
svr_rbf.fit(X_train, y_train)




svm_confidence = svr_rbf.score(X_test, y_test)
print("svm confidence: ", svm_confidence)



# svm_prediction = svr_rbf.predict(X_forecast)
# print(svm_prediction)

# Export the ML model
with open(r'stock_model.pickle', 'wb') as f:
    pickle.dump(svr_rbf, f)