import pandas as pd
import numpy as np
from keras.utils import np_utils
import tensorflow as tf
import matplotlib.pyplot as plt

# Load data
np.random.seed(0) #fix result of np.random
data = pd.read_csv(r"/home/natu/natu/NAL/IRIS/iris.data", sep = ',', header=None)
data = data.values # Shape: 150, 5 
np.random.shuffle(data)
y = data[:, data.shape[1] - 1] # Label - shape: 150, 1
X = data[:, 0 : data.shape[1] - 1].astype(float) # Data - shape: 150, 4
X_train = X[0:105, :] #shape: 120, 4
X_test = X[105:X.shape[0], :] #30, 4
y_train = y[0:105] #shape: 120, 4
y_test = y[105:y.shape[0]] #30, 4
del data, X, y

# Map label => 0, 1, 2 => one hot vector
classes = {'Iris-setosa' : 0,
           'Iris-versicolor' : 1,
           'Iris-virginica' : 2}
y_train = [classes[item] for item in y_train] 
y_test = [classes[item] for item in y_test] 
y_train = np.asarray(y_train)
y_test = np.asarray(y_test)
y_train = np_utils.to_categorical(y_train, 3)
y_test = np_utils.to_categorical(y_test, 3)

# Create place holder
def create_placeholders(X_train, y_train):
    X = tf.placeholder(tf.float32, [None, X_train.shape[1]])
    Y = tf.placeholder(tf.float32, [None, y_train.shape[1]])
    return X, Y

# Initialize parameters
def initialize_parameters(X_train, y_train):
    W = tf.get_variable("W", [X_train.shape[1], y_train.shape[1]], initializer = tf.contrib.layers.xavier_initializer(seed=1))
    b = tf.get_variable("b", [1, y_train.shape[1]], initializer = tf.zeros_initializer())
    parameters = {"W" : W,
                  "b" : b}
    return parameters

# forward propagation
def forward_propagation(X, parameters):
    W = parameters["W"]
    b = parameters["b"]
    Z = tf.add(tf.matmul(X, W), b)
    return Z

# compute cost
def compute_cost(Z, Y):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=Z, labels= Y))
    return cost

# softmax
def softmax(X_train, y_train, X_test, y_test, learning_rate = 0.1, num_epochs = 500, print_cost = True):
    tf.set_random_seed(0)
    X, Y = create_placeholders(X_train, y_train)
    parameters = initialize_parameters(X_train, y_train)
    Z = forward_propagation(X, parameters)
    cost = compute_cost(Z, Y)
    costs = []
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    init = tf.global_variables_initializer() # initialize all variables
    with tf.Session() as sess: # start session
        sess.run(init) # run initialization  
        for epoch in range(num_epochs): # training loop
            epoch_cost = 0 # cost related to a epoch
            _, cost1 = sess.run([optimizer, cost], feed_dict={X: X_train, Y: y_train})
            epoch_cost += cost1
            if print_cost == True and epoch % 10 == 0:
                print("Cost after epoch %i: %f" % (epoch, epoch_cost))
            if print_cost == True and epoch % 5 == 0:
                costs.append(epoch_cost)
        # save parameteres
        parameters = sess.run(parameters)
        # plot costs
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()
        # calculate correct predictions
        correct_prediction = tf.equal(tf.argmax(Z, 1), tf.argmax(Y, 1)) # argmax: Tim so thu tu co gia tri lon nhat THEO TRUC HOANH => Xs cao nhat.
        # calculate accuracy on test set
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print("Train Accuracy:", accuracy.eval({X: X_train, Y: y_train}))
        print("Test Accuracy:", accuracy.eval({X: X_test, Y: y_test}))
        return parameters

parameters = softmax(X_train, y_train, X_test, y_test)
#print(parameters)






