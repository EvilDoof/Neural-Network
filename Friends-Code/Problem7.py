# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 17:42:09 2020

@author: aditya ramkumar
"""

import pandas as pd
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import math

df = pd.read_excel('D:\\BITS\\Third Year\\Neural Networks and Fuzzy Logic\\Assignment 1\\data3.xlsx',header=None)

# Shuffle your dataset 
shuffle_df = df.sample(frac=1).to_numpy()

# Define a size for your train set 
train_size = int(0.6 * len(df))
traindata = shuffle_df[:,:4]
labels = shuffle_df[:,4]

# Normalizing the data
traindata = (traindata - np.mean(traindata))/np.std(traindata)

for i in range(len(labels)):
    if(labels[i]==1):
        labels[i] = 0
    else:
        labels[i] = 1

# Split your dataset 
x_train = traindata[:train_size]
x_test = traindata[train_size:]
y_train = labels[:train_size]
y_test = labels[train_size:]

#adding column of ones 
col1 = np.ones((len(x_train),1))
col2 = np.ones((len(x_test),1))
x_train = np.hstack((col1,x_train))
x_test = np.hstack((col2,x_test))
'''
x_train = x_train.T
x_test = x_test.T
y_train = y_train.T
y_test = y_test.T

def initialize_weights_and_bias(dimension):
    w = np.full((dimension,1),0)
    return w

def sigmoid(z):
    y_head = 1/(1+np.exp(-z))
    return y_head

def forward_backward_propagation(w,x_train,y_train):
    # forward propagation
    z = np.dot(w.T,x_train)
    y_head = sigmoid(z)
    loss = -y_train*np.log(y_head)-(1-y_train)*np.log(1-y_head)
    cost = (np.sum(loss))/x_train.shape[1]      # x_train.shape[1]  is for scaling
    # backward propagation
    derivative_weight = (np.dot(x_train,((y_head-y_train).T)))/x_train.shape[1] # x_train.shape[1]  is for scaling
    #derivative_bias = np.sum(y_head-y_train)/x_train.shape[1]                 # x_train.shape[1]  is for scaling
    gradients = {"derivative_weight": derivative_weight}
    return cost,gradients

def update(w, x_train, y_train, learning_rate,number_of_iterarion):
    cost_list = []
    cost_list2 = []
    W = []
    index = []
    # updating(learning) parameters is number_of_iterarion times
    for i in range(number_of_iterarion):
        # make forward and backward propagation and find cost and gradients
        cost,gradients = forward_backward_propagation(w,x_train,y_train)
        cost_list.append(cost)
        # lets update
        w = w - learning_rate * gradients["derivative_weight"]
        #b = b - learning_rate * gradients["derivative_bias"]
        W.append(w)
        if i % 10 == 0:
            cost_list2.append(cost)
            index.append(i)
            #print ("Cost after iteration %i: %f" %(i, cost))
    # we update(learn) parameters weights and bias
    parameters = {"weight": w}
    plt.plot(index,cost_list2)
    plt.xticks(index,rotation='vertical')
    plt.xlabel("Number of Iterarion")
    plt.ylabel("Cost")
    plt.show()
    return parameters, gradients, cost_list,W

def predict(w,x_test):
    # x_test is a input for forward propagation
    z = sigmoid(np.dot(w.T,x_test))
    Y_prediction = np.zeros((1,x_test.shape[1]))
    # if z is bigger than 0.5, our prediction is sign one (y_head=1),
    # if z is smaller than 0.5, our prediction is sign zero (y_head=0),
    for i in range(z.shape[1]):
        if z[0,i]<= 0.5:
            Y_prediction[0,i] = 0
        else:
            Y_prediction[0,i] = 1

    return Y_prediction

def logistic_regression(x_train, y_train, x_test, y_test, learning_rate ,  num_iterations):
    # initialize
    dimension =  x_train.shape[0]  # that is 4096
    w = initialize_weights_and_bias(dimension)
    # do not change learning rate
    parameters, gradients, cost_list,W = update(w, x_train, y_train, learning_rate,num_iterations)
    
    y_prediction_test = predict(parameters["weight"],x_test)
    y_prediction_train = predict(parameters["weight"],x_train)

    # Print train/test Errors
    print("train accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_train - y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_test - y_test)) * 100))
    return cost_list,gradients,W

learning_rate = 0.05
num_iterations = 100
cost, gradients,W = logistic_regression(x_train,y_train,x_test,y_test,learning_rate,num_iterations)
W = np.array(W)
'''


k = 100
alpha = 0.01
w = np.zeros((1,5))
W = np.zeros((k,5))
H = np.zeros((x_train.shape[0],1))
J = np.zeros((k,1))

    

def hyp(x):
    h = x@w.T
    for i in range(len(h)):
        H[i] = 1/(1+math.exp(-h[i]))
    return H
def errorsum(H,y):
    err = H.T-y
    return err.T
def evalcostfunc(x,y,w):
    H = hyp(x)
    L = np.log(H)
    M = np.log(1-H)
    j = y@L+(1-y)@M
    return j
for i in range(k):
    H = hyp(x_train)
    e = errorsum(H,y_train)
    G = np.transpose(x_train)@e
    w = w - alpha*G.T
    W[i] = w
    J[i] = evalcostfunc(x_train,y_train,w)
plt.plot(-J)
print(w)
#print(H,y_train)
P = hyp(x_test)
P = P[:40]
def mse(P,O):
    error = P.T-O
    error = np.square(error.T)
    s = np.sum(error)/len(P)
    return s
print("Training MSE: ",mse(x_train,y_train))
print("Test MSE: ",mse(P,y_test))
fig = plt.figure()
ax = plt.axes(projection='3d')
zline = J.flatten()
xline = W[:,1].flatten()
yline = W[:,2].flatten()
ax.plot3D(xline,yline,zline,'red')
acc=0
sen=0
pos=0
spe=0
neg=0
for i in range(len(P)):
    if(P[i]>0.5):
        P[i]=1
    else:
        P[i]=0
    if P[i]==y_test[i]:
        acc += 1
    if P[i]==1 and y_test[i]==1:
        sen += 1
    if y_test[i]==1:
        pos += 1
    if P[i]==0 and y_test[i]==0:
        spe += 1
    if y_test[i]==0:
        neg += 1
print("Accuracy = ",acc/len(y_test))
print("Sensitivity = ",sen/pos)
print("Specivicity = ",spe/neg)

        

    
            