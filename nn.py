import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.datasets import mnist
from sklearn.metrics import accuracy_score

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(X_train.shape[0], -1).T
X_test = X_test.reshape(X_test.shape[0], -1).T

m = len(X_train) + len(X_test)
n = X_train.shape[1] + 1

def init_parameters():
    W1 = np.random.randn(10, 784)  
    b1 = np.random.randn(10, 1) 
    W2 = np.random.randn(10, 10)
    b2 = np.random.randn(10, 1)
    return W1, b1, W2, b2

def relu(Z):
    return np.maximum(Z, 0)

def deriv_relu(Z):
    return Z > 0

def softMax(Z):
    return np.exp(Z) / np.sum(np.exp(Z), axis=0, keepdims=True)

def forwardProp(W1, b1, W2, b2, X):
    Z1 = np.dot(W1, X) + b1
    A1 = relu(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = softMax(Z2)
    
    return Z1, A1, Z2, A2.T

def oneHot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

def backProp(Z1, A1, Z2, A2, W2, X, Y):
    m = Y.size
    one_hot_Y = oneHot(Y)
    dZ2 = A2 - one_hot_Y
    dW2 = (1/m) * np.dot(dZ2, A1.T)
    db2 = (1/m) * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = np.dot(W2.T, dZ2) * deriv_relu(Z1)
    dW1 = (1/m) * np.dot(dZ1, X.T)
    db1 = (1/m) * np.sum(dZ1, axis=1, keepdims=True)
    return dW1, db1, dW2, db2

def update_parameters(w1, b1, w2, b2, dw1, db1, dw2, db2, alpha):
    w1 = w1 - alpha * dw1
    b1 = b1 - alpha * db1
    w2 = w2 - alpha * dw2
    b2 = b2 - alpha * db2
    
    return w1, b1, w2, b2

def gradientDescent(X, Y, iterations, alpha):
    W1, b1, W2, b2 = init_parameters()
    for i in range(iterations):
        Z1, A1, Z2, A2 = forwardProp(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = backProp(Z1, A1, Z2, A2, W2, X, Y)
        W1, b1, W2, b2 = update_parameters(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        if i % 10 == 0:
            print(f"Iteration: {i}")
            y_pred = np.argmax(A2, axis=0)
            acc = accuracy_score(y_pred, y_train)
            print(f"Accuracy: {acc}")
            
    return W1, b1, W2, b2

W1, b1, W2, b2 = gradientDescent(X_train, y_train, 500, 0.1)