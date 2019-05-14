import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelBinarizer
import tensorflow as tf
#from tensorflow import tf_utils
#from tf_utils import convert_to_one_hot
from keras.utils import to_categorical

# Load data
np.random.seed(0) #fix result of np.random
data = pd.read_csv(r"D:\NAL\IRIS\iris.data", sep = ',', header=None)
data = data.values # Shape: 150, 5 
np.random.shuffle(data)
y = data[:, data.shape[1] - 1] # Label - shape: 150, 1
X = data[:, 0 : data.shape[1] - 1] # Data - shape: 150, 4
X_train = X[0:105, :] #shape: 120, 4
X_test = X[105:X.shape[0], :] #30, 4
y_train = y[0:105] #shape: 120, 
y_test = y[105:y.shape[0]] #30, 
del data, X, y

# One-hot vector label
encoder = LabelBinarizer()
y_train = encoder.fit_transform(y_train) #shape: 120, 3
y_test = encoder.fit_transform(y_test) #shape: 30, 3

"""
# initialize parameters
def initialize_parameters (d, c): #d: dimension, c: classes
    W = np.random.randn(d, c) * 0.01
    b = np.zeros(shape = (1, c))
    parameters = {"W" : W, "b" : b}
    return parameters

# forward propagation
def forward_propagation (X_train, parameters):
    W = parameters["W"]
    b = parameters["b"]
    Z = np.dot(X_train, W) + b
    A = np.exp(Z.astype(float))/np.sum(np.exp(Z.astype(float)), axis= 1, keepdims = True) # Without keepdims: Shape cua mau (120,). with keepdims: Shape cua mau: (120,1)
    cache = {"Z" : Z, "A" : A}
    return A, cache

# Cost function
def compute_cost (A, y_train):
    n = X_train.shape[0]
    cost = -1/n * np.sum(np.sum(y_train * np.log(A), axis= 1), axis = 0) # Search softmax cost function
    return cost

# back propagation
def back_propagation (parameters, cache, X_train, y_train):
    n = X_train.shape[0]
    A = cache["A"]
    dZ = A - y_train
    dW = 1/n * np.dot(X_train.T, dZ)
    db = 1/n * np.sum(dZ, keepdims= True, axis= 0)
    grads = {"dW" : dW, "db" : db}
    return grads, dZ

# update parameters
def update_parameters (parameters, grads, learning_rate = 1/(10^4)):
    W = parameters["W"]
    b = parameters["b"]
    dW = grads["dW"]
    db = grads["db"]
    W = W - learning_rate * dW
    b = b - learning_rate * db
    parameters = {"W" : W, "b" : b}
    return parameters

# softmax regression
def softmax (X_train, y_train, iter = 500, print_cost=True):
    parameters = initialize_parameters(X_train.shape[1], y_train.shape[1])
    W = parameters["W"] 
    b = parameters["b"]
    # Loop
    for i in range(0, iter): 
        A, cache = forward_propagation(X_train, parameters)
        cost = compute_cost(A, y_train)
        grads, dZ = back_propagation (parameters, cache, X_train, y_train)
        parameters = update_parameters (parameters, grads)
        # Print cost
        if print_cost and i % 100 == 0:
            print("Cost after iteration %i: %f" % (i, cost))
    return parameters

parameters = softmax (X_train, y_train, print_cost=False)

# predict
def predict(X_test, parameters):
    A, cache = forward_propagation (X_test, parameters)
    predictions = np.round(A)
    return predictions
predictions = predict(X_test, parameters)

t = 0 # Number of true predict  
f = 0 # Number of false predict
for i in range(y_test.shape[0]):
    if (predictions[i] == y_test[i]).all():
        t += 1
    else:
        f += 1
accuracy = t/y_test.shape[0]
print(accuracy)
"""










