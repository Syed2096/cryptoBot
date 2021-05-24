#Import libraries
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader as web
import datetime as dt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from keras.models import load_model

#Stock Object
class Stock:
    def __init__(self, symbol):
        self.symbol = symbol
        self.alreadyHave = False
        self.prices = []
        self.priceBoughtAt = 0
        self.quantityBought = 0
        self.predictedPrices = []

stocks = []

#Open file
try:
    with open("crypto.txt", "rb") as filehandler:
        stocks = pickle.load(filehandler)

except:          
    print("No file")

for stock in stocks:
    #Load data
    company = stock.symbol
    
    #Prepare data
    scaler = MinMaxScaler(feature_range=(0, 1))
    #scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))
    prices = np.array(stock.prices)
    scaled_data = scaler.fit_transform(prices.reshape(-1, 1))

    #How far to look to determine price
    prediction_days = 800

    #How far in the future u want the price
    future_day = 120

    x_train = []
    y_train = []

    for x in range(prediction_days, len(scaled_data) - future_day):
        x_train.append(scaled_data[x - prediction_days:x, 0])
        y_train.append(scaled_data[x + future_day, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
        
    try:
        model = load_model("model.h5")

    except:    
        #Build model
        model = Sequential()

        #Experiment with layers, more layers longer time to train
        model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
        model.add(Dropout(0.2))
        model.add(LSTM(units=50, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(units=50))
        model.add(Dropout(0.2))
        model.add(Dense(units=1)) #Prediction of next closing value

        model.compile(optimizer='adam', loss='mean_squared_error')
        #Epoch = how many times model sees data, batchsize = how many units it sees at once

    model.fit(x_train, y_train, epochs=100, batch_size=100)
    
    #Test model accuracy on existing data
    #Load test data
    #test_start = dt.datetime(2021, 1, 1)
    #test_end = dt.datetime.now()
    #test_data = web.DataReader(company, 'yahoo', test_start, test_end)
    #actual_prices = test_data['Close'].values
    #total_dataset = pd.concat((data['Close'], test_data['Close']), axis=0)
    #total_dataset = stock.prices

    #model_inputs = np.array(total_dataset[len(total_dataset) - prediction_days:])
    #model_inputs = model_inputs.reshape(-1, 1)
    #model_inputs = scaler.transform(model_inputs)

    #Make prediction on test data
    #x_test = []

    #for x in range(prediction_days, len(model_inputs)):
    #    x_test.append(model_inputs[x - prediction_days:x, 0])

    #x_test = np.array(x_test)
    #x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    #predicted_prices = model.predict(x_test)
    #predicted_prices = scaler.inverse_transform(predicted_prices)

    #Plot test prediction
    
    #plt.plot(actual_prices, color='black', label=f"Actual {company} Price")
    #plt.plot(predicted_prices, color='green', label=f"Predicted {company} Price")
    #plt.title(f"{company} Share Price")
    #plt.xlabel("Time")
    #plt.ylabel(f"{company} Share Price")
    #plt.legend()
    #plt.show()

    #Prediction
    #real_data = [model_inputs[len(model_inputs) + 1 - prediction_days:len(model_inputs + 1), 0]]
    #real_data = np.array(real_data)
    #real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))

    #prediction = model.predict(real_data)
    #prediction = scaler.inverse_transform(prediction)
    #print(f"Prediction for {stock.symbol}: {prediction}")

    model.save('model.h5')
