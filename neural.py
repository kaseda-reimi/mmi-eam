import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import optimizers
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import os
import function as fc

input_size = 4
output_size = 2
hidden_size = 30

n_splits = 3
test_size = 0.1

lr = 0.001
momentum = 0.9
epochs = 10000
workers = os.cpu_count()

model_path =os.getcwd()+'/model'

def create_model():
    model = Sequential()
    model.add(Dense(hidden_size, input_dim = input_size, activation = "relu"))
    model.add(Dense(hidden_size, activation = "relu"))
    model.add(Dense(output_size, activation = "linear"))
    sgd = optimizers.Adam(learning_rate=lr)
    model.compile(loss = "mean_squared_error", optimizer = sgd, metrics = ["mae"])
    return model

def main_cross_validation():
    data = fc.get_data('data.txt')
    X = data[:, :input_size]
    Y = data[:, input_size:]
    N = X.shape[0]

    X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=test_size)
    kf = KFold(n_splits = n_splits)
    _history = []

    for train_index, val_index in kf.split(X_train, Y_train):
        model = create_model()
        #model.summary()
        model.fit(X_train[train_index], Y_train[train_index], epochs=epochs, workers=workers, use_multiprocessing=True)
        _history.append(model.evaluate(x=X_train[val_index], y=Y_train[val_index]))
    
    predict = model.predict(X_test)

    error = abs(predict - Y_test)
    error_sum = 0
    for i in range(predict.shape[0]):
        print(Y_test[i,:])
        print(predict[i,:])
        print("")
        error_sum += error[i]*error[i]
    print(error_sum/predict.shape[0])
    print(_history)
    
    model.save(model_path)

def main():
    data = fc.get_data('data.txt')
    X = data[:, :input_size]
    Y = data[:, input_size:]
    N = X.shape[0]

    X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=test_size)
    model = create_model()
    model.fit(X_train, Y_train, epochs=epochs, workers=workers, use_multiprocessing=True)
    predict = model.predict(X_test)
    error = abs(predict - Y_test)
    error_sum = 0
    for i in range(predict.shape[0]):
        print(Y_test[i,:])
        print(predict[i,:])
        print("")
        error_sum += error[i]*error[i]
    print(error_sum/predict.shape[0])
    model.save(model_path)
    fc.write_data('/Y.txt', Y_test)
    fc.write_data('/predict.txt', predict)


if __name__ == '__main__':
    main()

