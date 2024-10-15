import numpy as np
import matplotlib.pyplot as plt
from utils import *
import copy
import math
from keras.models import Sequential
import tensorflow as tf
from keras.layers import Dense
sizeA = 1024
num_f = 1
population = np.random.rand(sizeA,num_f)
earnings = np.random.rand(sizeA,num_f)
np.save("data/x_train.npy", population)
np.save("data/y_train.npy", earnings)
population = np.random.rand(sizeA,num_f)
earnings = np.random.rand(sizeA,num_f)
np.save("data/x_train.npy", population)
np.save("data/y_train.npy", earnings)

def load_data(string1,string2):
    x_train = np.load(string1)
    y_train = np.load(string2)
    return x_train, y_train

def initialize_weights_and_bias(dimension):
    w = np.full((dimension, 1), 0.01)
    b = 0.0
    return w, b

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def compute_cost(w, b, x_train, y_train):
    z = np.dot(w.T, x_train) + b
    y_head = sigmoid(z)
    loss = -y_train * np.log(y_head) - (1 - y_train) * np.log(1 - y_head)
    cost = (np.sum(loss)) / x_train.shape[1]  # x_train.shape[1]  is for scaling
    return cost


def gradient_descent(x_train, y_train, learning_rate, num_iterations):
    m = len(x_train)
    theta = np.zeros((x_train.shape[0], 1))
    cost_history = []

    for i in range(num_iterations):
        z = np.dot(theta.T, x_train)  # Compute the linear regression
        y_head = sigmoid(z)  # Apply the sigmoid function
        loss = -y_train * np.log(y_head) - (1 - y_train) * np.log(1 - y_head)  # Compute the loss
        cost = np.sum(loss) / m  # Compute the cost
        cost_history.append(cost)  # Store the cost in each iteration

        gradient = (1 / m) * np.dot(x_train, (y_head - y_train).T)  # Compute the gradient
        theta = theta - learning_rate * gradient  # Update the parameters

        if i % 100 == 0:
            print(f"Cost after iteration {i}: {cost}")

    return theta, cost_history


def predict(theta,x_test):
    z = sigmoid(np.dot(x_test.T, theta[0].T))
    Y_prediction = np.zeros((1, x_test.shape[0]))
    for i in range(z.shape[1]):
        if z[0, i] <= 0.5:

            Y_prediction[0, i] = 0
        else:
            Y_prediction[0,i] = 1

    return Y_prediction
def logistic_regression(x_train,y_train,x_test,y_test,learning_rate, num_iterations):
    dimension = x_train.shape[0]
    w, b = initialize_weights_and_bias(dimension)
    parameters = gradient_descent(x_train, y_train, learning_rate, num_iterations)
    y_prediction_test = predict(parameters, x_test)
    y_prediction_train = predict(parameters, x_train)
    print("test accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_test - y_test)) * 100))
    print("train accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_train - y_train)) * 100))
    return parameters
x_train,y_train=load_data("data/x_train.npy","data/y_train.npy")
x_test, y_test = load_data("data/x_test.npy", "data/y_test.npy")

logistic_regression(x_train, y_train, x_test, y_test, 0.01, 1000)


def initialize_weights_and_bias(x_train, y_train):
    parameters ={
        "weight1" : np.random.randn(3,x_train.shape[0])*0.01,
        "bias1"   : np.zeros((3,1)),
        "weight2" : np.random.randn(y_train.shape[0],3)*0.01, 
        "bias2"   : np.zeros((y_train.shape[0],1))
    }
    return parameters

def compute_cost_NN(A2, Y, parameters):
    logprobs = np.multiply(np.log(A2), Y)
    cost = -np.sum(logprobs) / Y.shape[1]
    return cost

def forwar_backward_propagation_update_NN(parameters, X, Y,learning_rate):
    Z1 = np.dot(parameters["weight1"], x_train) + parameters["bias1"]
    A1 = np.tanh(Z1)
    Z2 = np.dot(parameters["weight2"], A1) + parameters["bias2"]
    A2 = sigmoid(Z2)

    cache = {"Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2}

    dZ2 = cache["A2"] - Y
    dW2 = np.dot(dZ2, cache["A1"].T) / X.shape[1]
    db2 = np.sum(dZ2, axis=1, keepdims=True) / X.shape[1]
    dZ1 = np.dot(parameters["weight2"].T, dZ2) * (1 - np.power(cache["A1"], 2))
    dW1 = np.dot(dZ1, X.T) / X.shape[1]
    db1 = np.sum(dZ1, axis=1, keepdims=True) / X.shape[1]
    
    grads = {"dweight1": dW1, "dbias1": db1, "dweight2": dW2, "dbias2": db2}
    
    parameters = {
            "weight1": parameters["weight1"] - learning_rate * grads["dweight1"],
            "bias1": parameters["bias1"] - learning_rate * grads["dbias1"],
            "weight2": parameters["weight2"] - learning_rate * grads["dweight2"],
            "bias2": parameters["bias2"] - learning_rate * grads["dbias2"],
        }

    return A2 ,parameters


def predict_NN(parameters, x_test):
    # x_test is a input for forward propagation
    A2,parameters = forwar_backward_propagation_update_NN(parameters,x_test,y_test,0.01)
    Y_prediction = np.zeros((1, x_test.shape[1]))
    # if z is bigger than 0.5, our prediction is sign one (y_head=1),
    # if z is smaller than 0.5, our prediction is sign zero (y_head=0),
    for i in range(A2.shape[1]):
        if A2[0, i] <= 0.5:
            Y_prediction[0, i] = 0
        else:
            Y_prediction[0, i] = 1

    return Y_prediction


def two_layer_neural_network(x_train, y_train, x_test, y_test, num_iterations):
    cost_list = []
    index_list = []
    # initialize parameters and layer sizes
    parameters = initialize_weights_and_bias(x_train, y_train)

    for i in range(0, num_iterations):
        # forward propagation

        A2,parameters = forwar_backward_propagation_update_NN(parameters,x_train,y_train,0.01 )
        cost=compute_cost_NN(A2, y_train, parameters)
        if i % 100 == 0:
            cost_list.append(cost)
            index_list.append(i)
            print("Cost after iteration %i: %f" % (i, cost))
    plt.plot(index_list, cost_list)
    plt.xticks(index_list, rotation="vertical")
    plt.xlabel("Number of Iterarion")
    plt.ylabel("Cost")
    plt.show()

    # predict
    y_prediction_test = predict_NN(parameters, x_test)
    y_prediction_train = predict_NN(parameters, x_train)

    # Print train/test Errors
    print(
        "train accuracy: {} %".format(
            100 - np.mean(np.abs(y_prediction_train - y_train)) * 100
        )
    )
    print(
        "test accuracy: {} %".format(
            100 - np.mean(np.abs(y_prediction_test - y_test)) * 100
        )
    )
    return parameters


parameters = two_layer_neural_network(
    x_train, y_train, x_test, y_test, num_iterations=500
)
