import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout


def get_training():
    dataset_train = pd.read_csv('NSE-TATAGLOBAL.csv')
    training_set = dataset_train.iloc[:, 1:2].values
    return training_set,dataset_train


def get_test():
    dataset_test = pd.read_csv('tatatest.csv')
    real_stock_price = dataset_test.iloc[:, 1:2].values
    return real_stock_price,dataset_test


def scale_data(training_set):
    sc = MinMaxScaler(feature_range = (0, 1))
    training_set_scaled = sc.fit_transform(training_set)
    return training_set_scaled,sc

def process_training_set(training_set_scaled):
    X_train = []
    y_train = []
    for i in range(60, 750):
        X_train.append(training_set_scaled[i-60:i, 0])
        y_train.append(training_set_scaled[i, 0])
    X_train, y_train = np.array(X_train), np.array(y_train)

    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    return X_train,y_train

def process_test_set(dataset_test,dataset_train,sc,model):
    dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0)
    inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
    inputs = inputs.reshape(-1,1)
    inputs = sc.transform(inputs)
    X_test = []
    for i in range(60, 76):
        X_test.append(inputs[i-60:i, 0])
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    predicted_stock_price = model.predict(X_test)
    predicted_stock_price = sc.inverse_transform(predicted_stock_price)
    return predicted_stock_price


def keras_model(X_train):
    model = Sequential()

    model.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
    model.add(Dropout(0.2))

    model.add(LSTM(units = 50, return_sequences = True))
    model.add(Dropout(0.2))

    model.add(LSTM(units = 50, return_sequences = True))
    model.add(Dropout(0.2))

    model.add(LSTM(units = 50))
    model.add(Dropout(0.2))

    model.add(Dense(units = 1))

    return model


def plot_data(real_stock_price,predicted_stock_price):
    plt.plot(real_stock_price, color = 'black', label = 'TATA Stock Price')
    plt.plot(predicted_stock_price, color = 'green', label = 'Predicted TATA Stock Price')
    plt.title('TATA Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('TATA Stock Price')
    plt.legend()
    plt.show()

def main():
    training_set, dataset_train = get_training()
    real_stock_price, dataset_test = get_test()

    training_set_scaled,sc = scale_data(training_set)
    #
    X_train, Y_train = process_training_set(training_set_scaled)
    model = keras_model(X_train)

    model.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics=['accuracy'])
    model.fit(X_train, Y_train, epochs = 20, batch_size = 32)
    predicted_stock_price = process_test_set(dataset_test,dataset_train,sc,model)
    plot_data(real_stock_price,predicted_stock_price)

main()
