'''
Created on 15.03.2021

@author: diana.tahchieva
'''
# -*- coding: utf-8 -*-

import numpy as np
from random import randrange
import matplotlib.pyplot as plt


# function that calculates the Gaussian kernel. 
# Here it is calculated explicitly for tutorial purposes.
 
def gaussian_kernel(X, sigma, llambda):
    pairwise_dists = np.zeros((np.shape(X)[0], np.shape(X)[0]))
    for i in range(len(X)):
        for j in range(len(X)):
            pairwise_dists[i, j] = np.sqrt((X[i] - X[j]) ** 2)
    K = np.exp(-pairwise_dists ** 2 / 2 * (sigma ** 2))
    
    """add regularisation"""
    K[np.diag_indices(np.shape(X)[0], 2)] -= llambda
    return K


def train(X, Y,sigma, llambda):
    kernel = gaussian_kernel(X,sigma, llambda)
    #Solve a linear matrix equation, or system of linear scalar equations.
    #i.e., full rank, linear matrix equation `ax = b`
    weights = np.linalg.solve(kernel, Y) 
    return weights


def predict(trainX, predictX, weights, sigma):
    #(x - y_pred)^2 / 2*sigma^2
    K = np.exp(-(trainX - predictX) ** 2 / 2 * (sigma ** 2))
    return np.dot(weights, K)
    

# Split a dataset into k folds
def cross_validation_split(dataset, folds=3):
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / folds)
    for i in range(folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split



trainX = np.linspace(-5, 5, 50)
trainY = trainX ** 2

"""
exercise
try more complex functions (e.g. polinomial)
"""
predictX = np.array([2.5])
predictY = predictX**2

dataset = np.column_stack((trainX, trainY))
folds = cross_validation_split(dataset, folds=3)
#print(folds[0])
#print([i[0] for i in folds[0]])

sigma=0.01
llambda=1e-12

#print(gaussian_kernel(trainX,sigma, llambda))
accuracy = []

for fold in folds:
    X_fold = ([i[0] for i in fold])
    Y_fold = trainig_fold = ([i[1] for i in fold])
    
    weights = train(X_fold, Y_fold,sigma, llambda)
    
    y_pred = (predict(X_fold, predictX, weights, sigma))
    print("predicted y ",y_pred)
    print ("true y", predictY)
    accuracy = predictY - y_pred
    
print("model accuracy Mean absolute error", np.mean(abs(accuracy)))