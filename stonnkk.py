#Importing the necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout


#Get the Data
data = pd.read_csv('RelianceTrain.csv')
data = data.fillna(data.mean())
X = data.iloc[:, [5]].values

# # Splitting the dataset into the Training set and Test set
# from sklearn.model_selection import train_test_split
# X_train, X_test = train_test_split(X, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(X)


# Creating a data structure with 60 timesteps and 1 output
X_train1 = []
y_train1 = []
for i in range(60, training_set_scaled.shape[0]):
    X_train1.append(training_set_scaled[i-60:i, 0])
    y_train1.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train1), np.array(y_train1)

# Reshaping for LSTM 
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))


#Initialize the RNN
model = Sequential()

#Adding first LSTM layer
model.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
model.add(Dropout(0.2))

#Adding second LSTM layer
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))

#Adding third LSTM layer
model.add(LSTM(units= 50, return_sequences=True))
model.add(Dropout(0.2))

#Adding fourth LSTM layer
model.add(LSTM(units=50, return_sequences=False))
model.add(Dropout(0.2))

#Adding Output layer
model.add(Dense(units=1))

#Compiling the RNN
model.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics = ['mean_absolute_error'])

#Fitiing the RNN
model.fit(X_train, y_train, epochs = 100, batch_size = 25)

#Test Data
data_test = pd.read_csv('RelianceTest.csv')
XX = data_test.iloc[:, [5]].values

#Prediction
data_total = pd.concat((data['Adj Close'], data_test['Adj Close']), axis = 0)
inputs = data_total[len(data_total) - len(data_test) - 60:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)


X_test = []
for i in range(60, inputs.shape[0]):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = model.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

# Visualising the results
plt.plot(XX, color = 'red', label = 'Actual Reliance Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Reliance Stock Price')
plt.title('Reliance Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()






