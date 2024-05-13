import numpy as np
import pandas as pd

def init_params():
    w1 = np.random.randn(32, 784) * np.sqrt(1/784) # 32, 784
    b1 = np.zeros((32, 1))
    w2 = np.random.randn(10, 32) * np.sqrt(1/784) #10, 32
    b2 = np.zeros((10, 1))
    return w1, b1, w2, b2

def leaky_relu(z):
    return np.where(z > 0, z, .01 * z)

def leaky_relu_deriv(z):
    return np.where(z > 0, 1, .01)

def softmax(z):
    eps = 1e-10
    return np.exp(z+eps) / np.sum(np.exp(z)+ eps, axis = 0, keepdims = True)

def forward(x, w1, b1, w2, b2):
    z1 = np.dot(w1, x) + b1 #32, 60000
    a1 = leaky_relu(z1) #32, 60000
    z2 = np.dot(w2, a1) + b2 #10, 60000
    a2 = softmax(z2) #  10 , 60000
    return z1, a1, z2, a2

def one_hot(y):
    one_hot_y = np.zeros((np.max(y) + 1, y.size)) #10, 60000
    one_hot_y[y, np.arange(y.size)] = 1
    return one_hot_y # 10 , 60000

def cat_cross(one_hot_y, a):
    eps = 1e-10
    l = - np.sum(one_hot_y * np.log(a + eps)) / 60000
    return l

def pred(a2):
    prediction = np.argmax(a2, axis = 0, keepdims=True)
    return prediction

def acc(a2, y):
    prediction = pred(a2)
    accuracy = np.sum(prediction == y) / 60000 * 100
    return accuracy


def backward(x, one_hot_y, w2, a2, a1, z1):
    dz2= a2 - one_hot_y
    dw2 = np.dot(dz2, a1.T) / 60000
    db2 = np.sum(dz2, axis=1, keepdims=True) / 60000
    dz1 = np.dot(w2.T, dz2) * leaky_relu_deriv(z1)
    dw1 = np.dot(dz1, x.T) / 60000
    db1 = np.sum(dz1, axis = 1, keepdims=True) / 60000
    return dw1, db1, dw2, db2

def update(w1, b1, w2, b2, dw1, db1, dw2, db2, alpha):
    w1 = w1 - alpha * dw1
    b1 = b1 - alpha * db1
    w2 = w2 - alpha * dw2
    b2 = b2 - alpha * db2
    return w1, b1, w2, b2

def gradient_descent(x, y, w1, b1, w2, b2, epochs, alpha):
    one_hot_y = one_hot(y)
    for epoch in range(epochs):
        z1, a1, z2, a2 = forward(x, w1, b1, w2, b2)

        l = cat_cross(one_hot_y, a2)
        accuracy = acc(a2, y)

        dw1, db1, dw2, db2 = backward(x, one_hot_y, w2, a2, a1, z1)
        w1, b1, w2, b2 = update(w1, b1, w2, b2, dw1, db1, dw2, db2, alpha)

        print(f"Epoch: {epoch}")
        print(f"Accuracy: {accuracy}%")
        print(f"Loss: {l}\n")
    return w1, b1, w2, b2

def model(x, y, epochs, alpha):
    w1, b1, w2, b2 = init_params()
    w1, b1, w2, b2 = gradient_descent(x, y, w1, b1, w2, b2, epochs, alpha)
    return w1, b1, w2, b2



if __name__ == "__main__":
    X_train = np.load('data/kmnist-train-imgs.npz')['arr_0'].reshape(-1, 60000) / 255 #  784, 60000
    Y_train = np.load('data/kmnist-train-labels.npz')['arr_0'].reshape(1, -1)

    model(X_train, Y_train, 1000, .1)
