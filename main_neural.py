import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import optimizers
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import function as fc

input_size = 4
output_size = 2

if __name__ == '__main__':
    data = fc.get_data()
    X = data[:, :input_size]
    Y = data[:, input_size:]

    hidden_size = 9
    N = X.shape[0]

    X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.1)
    kf = KFold(n_splits = 4)
    _history = []
    for train_index, val_index in kf.split(X_train, Y_train):
        model = Sequential()
        model.add(Dense(hidden_size, input_dim = input_size, activation = "relu"))
        model.add(Dense(hidden_size, activation = "relu"))
        model.add(Dense(output_size, activation = "linear"))
        sgd = optimizers.SGD(lr = 0.001, momentum = 0.9)
        model.compile(loss = "mean_squared_error", optimizer = sgd, metrics = ["mae"])

        #model.summary()

        model.fit(X_train[train_index], Y_train[train_index], epochs=10000)
        _history.append(model.evaluate(x=X_train[val_index], y=Y_train[val_index]))



    predict = model.predict(X_test)

    E = abs(predict - Y_test)
    E_sum = 0

    for i in range(predict.shape[0]):
        #print(X_test[i, :])
        print(Y_test[i,:])
        print(predict[i,:])
        print("")
        E_sum += E[i]*E[i]

    print(E_sum/predict.shape[0])
    print(_history)
    
    model.save('mmi-eam/model')
    model.save_weights('mmi-eam/weight')
   
    
