# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 22:29:53 2020

@author: aditya ramkumar
"""


import pandas as pd
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

#reading the values into the variables
X_train = pd.read_excel('D:\\BITS\\Third Year\\Neural Networks and Fuzzy Logic\\Assignment 1\\training_feature_matrix.xlsx',header=None).to_numpy()
y_train = pd.read_excel('D:\\BITS\\Third Year\\Neural Networks and Fuzzy Logic\\Assignment 1\\training_output.xlsx',header=None).to_numpy()
X_test = pd.read_excel('D:\\BITS\\Third Year\\Neural Networks and Fuzzy Logic\\Assignment 1\\test_feature_matrix.xlsx',header=None).to_numpy()
y_test = pd.read_excel('D:\\BITS\\Third Year\\Neural Networks and Fuzzy Logic\\Assignment 1\\test_output.xlsx',header=None).to_numpy()

#normalization
mean1 = X_train.mean(axis=0)
std1 = X_train.std(axis=0)
X_train = (X_train -mean1)/std1
X_test = (X_test -mean1)/std1

mean2 = y_train.mean(axis=0)
std2 = y_train.std(axis=0)
y_train = (y_train -mean2)/std2
#y_test = (y_test -mean2)/std2

#adding column of ones 
col1 = np.ones((len(X_train),1))
col2 = np.ones((len(X_test),1))
X_train = np.hstack((col1,X_train))
X_test = np.hstack((col2,X_test))

w = np.zeros((1,3))
#learning rate
alpha = 0.001
#no. of iterations
t=20
#mini batch
m=50

def evalcostfuc(x,y,w):
    j=0
    for i in range(m):
        j += (np.dot(w,x[i].T)-y[i])**2
    j = 0.5*j
    return j

#initializing J and W as zero column matrix
J=np.zeros((t*m,1))
W=np.zeros((t*m,3))
h=0
e=0
a=np.arange(0,len(X_train))
for i in range(t):
    np.random.permutation(a)
    b=a[0:50]
    for j in range(m):
        for k in range(3):
            w[0,k] = w[0,k] - alpha*(np.dot(w,X_train[a[j]].T)-y_train[a[j]])*X_train[a[j],k]
        J[h] = evalcostfuc(X_train,y_train,w)
        h += 1
        W[e] = w
        e += 1
plt.plot(J)
T = np.zeros((len(X_train)))
for i in range(len(X_train)):
    T[i] = np.dot(w,X_train[i].T)

MSE = 0
for i in range(len(X_train)):
    diff = T[i]-y_train[i]
    sq_dif = diff**2
    MSE += sq_dif
MSE = MSE/len(X_train)
print("The MSE on the Training set = ",MSE)

MSE_test = 0
for i in range(len(X_test)):
    diff = ((T[i]*std2)+mean2)-y_test[i]
    sq_dif = diff**2
    MSE_test += sq_dif
MSE_test = MSE_test/len(X_test)
print("The MSE on the Test set = ",MSE_test)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

a =W[:,1]
b =W[:,2]
c =J

ax.scatter(a, b, c, c='r', marker='o')

ax.set_xlabel('W1')
ax.set_ylabel('W2')
ax.set_zlabel('Cost')

plt.show()







