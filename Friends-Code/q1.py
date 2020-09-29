# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 18:56:00 2020

@author: DELL
"""

# import modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# loading data
training_df = pd.read_excel("training_feature_matrix.xlsx", header=None)
X = training_df.to_numpy()
train_op_df = pd.read_excel("training_output.xlsx", header=None)
Y = train_op_df.to_numpy()
testing_df = pd.read_excel("test_feature_matrix.xlsx", header=None)
X_test = testing_df.to_numpy()
test_op_df = pd.read_excel("test_output.xlsx", header=None)
Y_test = test_op_df.to_numpy()

# normalizing
def normalize(features):
    """min-max normalization of provided feature matrix

    Args:
        features (matrix): a matrix of input parameters x1, x2, ... xn

    Returns:
        None: None
    """
    fmin = np.min(features)
    frange = np.max(features, axis=0) - np.min(features, axis=0)

    features -= fmin

    features /= frange

    return features

x_training_normalized = normalize(X)

y_train = normalize(Y)

x_testing_normalized = normalize(X_test)

y_test = normalize(Y_test)

# bias term for feature matrices
m_train = x_training_normalized.shape[0]
m_test = x_testing_normalized.shape[0]
x_train = np.insert(x_training_normalized, 0, np.ones(m_train), axis=1)
x_test = np.insert(x_testing_normalized, 0, np.ones(m_test), axis=1)

def hypothesis(theta, X, n):
    """hypothesis function makes prediction with given weights

    Args:
        theta ([float]): an array of weights; size n
        X ([[float]]): feature matrix for making predictions
        n (int): number of weight values

    Returns:
        [float]: array of predicted output values for calculating MSE
    """
    h = np.zeros([X.shape[0], 1])
    for i in range(X.shape[0]):
        for j in range(n):
            h[i] += theta[j] * X[i][j]
    return h

def grad_desc(theta, alpha, num_iters, h, X, Y, n, thetalist):
    """batch gradient descent function

    Args:
        theta ([float]): an array of weights; size n
        alpha (float): learning rate; lies between 0 and 1
        num_iters (int): number of iterations
        h ([float]): predicted output matrix
        X ([[float]]): feature matrix
        Y ([[float]]): output feature matrix
        n (int): size of weight matrix
        thetalist ([float]): theta history 

    Returns:
        [float], [float]: final weight values, cost history
    """
    cost_history = np.ones(num_iters)
    for i in range(num_iters):
        for j in range(n):
            for k in range(X.shape[0]):
                theta[j] -= alpha*(h[k]-Y[k])*(X[k][j])
        h = hypothesis(theta, X, n)
        for k in range(X.shape[0]):
            cost_history[i] += 0.5*(h[k]-Y[k])**2
        thetalist.append(list(theta))
        if i%20 == 0:
            print("iteration", i, ": ", "theta[0]", theta[0], '\n',  "theta[1]",  theta[1], '\n', "theta[2]", theta[2])
    return theta, cost_history

def linear_regression(X, Y, alpha, num_iters, thetalist):
    """linear regression function

    Args:
        X ([[float]]): input feature matrix
        Y ([[float]]): output matrix
        alpha (float): learning rate; lies between 0 and 1
        num_iters (int): number of iterations
        thetalist ([float]): theta history

    Returns:
        [float], [float]: final weight values, cost history
    """
    n = X.shape[1]

    theta = np.zeros(n)

    h = hypothesis(theta, X, n)
    theta, cost = grad_desc(theta, alpha, num_iters, h, X, Y, n, thetalist)
    return theta, cost

# implementation
alpha = 0.0001
num_iters = 80
thetalist = []
theta, cost = linear_regression(x_train, y_train, alpha, num_iters, thetalist)

# cost vs iterations plot
n_iterations = [x for x in range(1, num_iters+1)]
plt.plot(n_iterations, cost)
plt.xlabel('iterations')
plt.ylabel('cost')

# 3D plot
# cost vs  theta[1]    and     theta[2]
#   Z  vs     X        and        Y
thetaX = []
thetaY = []
for i in thetalist:
    thetaX.append(i[1])
    thetaY.append(i[2])
# 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(thetaX, thetaY, cost)
plt.xlabel("theta1")
plt.ylabel("theta2")

# printing training and testing mse values
y_predicted = hypothesis(theta, x_train, 3)
mse_training = np.average((y_predicted - y_train)**2)
print("training MSE:")
print(mse_training)
y_predicted = hypothesis(theta, x_test, 3)
mse_testing = np.average((y_predicted - y_test)**2)
print("test MSE:")
print(mse_testing)