import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from neural import model_path


input_size = 4
lr = 0.001
mom = 0.9
epoch = 5000
pop = 50
#w1 = 0.56

def individuals():
    ind = np.random.rand(pop, input_size)
    #ind[:,0] = w1
    return ind


def calc(x):
    #x_model[:,1] = x[:,1] * 0.5 + 0.3
    #x_model[:,3] = x[:,3] * (1 - x[:,2])
    x_model = tf.Variable(x)
    with tf.GradientTape() as tape:
        tape.watch(x_model)
        y = model(x_model, training=False) #[消光比, 無電界時の光出力]
        E = tf.reduce_sum(y,1)
    grad = tape.gradient(E, x_model)
    E = E.numpy()
    grad = grad.numpy()
    grad[:,0] = grad[:,0] - 4       #
    #grad[:,1] = grad[:,1] * 0.5
    #grad[:,3] = grad[:,3] * (1 - x[:,2])
    return y, E, grad



if __name__ == '__main__':
    model = load_model(model_path)

    momentum = tf.zeros([pop,input_size])
    x = individuals()
    y, E, grad = calc(x)
    #print(grad)
    #print(y)

    for i in range(epoch):
        _x = x
        x[:] = x[:] + lr * np.random.rand(1,input_size) * grad[:] + mom * momentum[:]
        x = np.where(x<0, 0, x)
        x = np.where(x>1, 1, x)
        momentum = x - _x

        y, E, grad = calc(x)
        #print(E.numpy()[0])
        #print(y[0])

    max_id = np.argmax(E)
    
    y = y.numpy()
    print(x[max_id])
    print(y[max_id])
    print(E[max_id])

    
