import numpy as np
import matplotlib.pyplot as plt
from utils import *
import copy
import math
from keras.models import Sequential
import tensorflow as tf
from keras.layers import Dense


# Load the data
def load_data():
    x_train = np.load("data/x_train.npy")
    y_train = np.load("data/y_train.npy")
    return x_train, y_train
x_test = np.load("data/x_test.npy")

pop_size = 97
profit_size = 97
num_features = 1
population = np.random.rand(pop_size, num_features)
earnings = np.random.rand(profit_size, num_features)
np.save("data/x_train.npy", population)
np.save("data/y_train.npy", earnings)

x_train, y_train = load_data()
print("The shape of x_train is:", x_train.size)
print("The shape of y_train is: ", y_train.size)
print("Number of training examples (m):", len(x_train))


def compute_cost(x_train, y_train, theta):
    m = len(x_train)
    h = np.dot(x_train, theta)
    cost = (1 / (2 * m)) * np.sum(np.square(h - y_train))
    return cost


def gradient_descent(x_train, y_train, learning_rate, num_iterations):
    m = len(x_train)
    theta = np.zeros((2, 1))
    cost = compute_cost(x_train, y_train, theta)
    while(cost > 0.042):
        cost = compute_cost(x_train, y_train, theta)
        h = np.dot(x_train, theta)
        gradient = (1 / m) * np.dot(x_train.T, (h - y_train))
        theta = theta - learning_rate * gradient
        print(f"Cost after iteration: {cost}")
        print(f"Theta after iteration: {theta[0]} , {theta[1]}")

    return theta


learning_rate = 0.001
num_iterations = 100

# Add a column of ones to x_train for the bias term
x_train = np.c_[np.ones((len(x_train), 1)), x_train]

theta = gradient_descent(x_train, y_train, learning_rate, num_iterations)

print("Learned parameters theta:", theta)
print(theta, "\n")

m = x_train.shape[0]
predicted = np.dot(x_train, theta)

print("Deneme ibo", np.dot(5, theta))

print(x_train.shape)
print(theta.shape)
print(predicted.shape)

# print(predicted[:10])


# plt.scatter(x_train[:, 1], y_train, marker="x", c="r")
# plt.plot(x_train[:, 1], predicted, label="Linear regression (Gradient descent)")

# # Set the title
# plt.title("Profits vs. Population per city")
# # Set the y-axis label
# plt.ylabel("Profit in $10,000")
# # Set the x-axis label
# plt.xlabel("Population of City in 10,000s")
# # Add a legend
# plt.legend()

# plt.show()
population = np.random.rand(100, 25)
earnings = np.random.rand(100, 25)
popnew=np.random.rand(100, 25)
np.save("data/new_x_train.npy", popnew)
np.save("data/x_train.npy", population)
np.save("data/y_train.npy", earnings)
layer_1 = Dense(units=25, activation="sigmoid")
layer_2 = Dense(units=15, activation="sigmoid")
layer_3 = Dense(units=1, activation="sigmoid")

model = Sequential([layer_1, layer_2, layer_3])

model.compile(loss="mean_squared_error", optimizer="adam", metrics=["accuracy"])

x_train, y_train = load_data()
print("The shape of x_train is:", x_train.size)
print("The shape of y_train is: ", y_train.size)
print("Number of training examples (m):", len(x_train))

model.fit(x_train, y_train)
model.predict(x_test)
model.evaluate(x_train, y_train)


# print("Model weights:", model.get_weights())
print("Model summary:", model.summary(), "\n")
