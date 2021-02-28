#!/usr/bin/env python
# coding: utf-8

# In[276]:


import numpy as np
import h5py
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')

def load_data():
    train_dataset = h5py.File('./dataset/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

    test_dataset = h5py.File('./dataset/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

    classes = np.array(test_dataset["list_classes"][:]) # the list of classes
    
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes

np.random.seed(1)


# In[277]:


train_x_orig, train_y, test_x_orig, test_y, classes = load_data()


# In[278]:


train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T
test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T

train_x_flatten = train_x_flatten/255
test_x_flatten = test_x_flatten/255


# In[279]:


def initialiseParams(n_x):
    W = np.random.randn(n_x, 1)*0.01
    b = 0
    
    return W, b


# In[280]:


def sigmoid(Z):
    return 1/(1+np.exp(-Z))


# In[303]:


def Prop(X, W, b, Y, i):
    n_x = X.shape[0]
    m = X.shape[1]
    
    Z = np.dot(W.T, X) + b
    A = sigmoid(Z)
    

    cost = -np.sum(np.dot(Y, np.log(A).T) + np.dot(1-Y, np.log(1-A).T))/m

    AY = A-Y
    dW = np.dot(X, AY.T)/m
    db = np.sum(AY)/m
    grads = {'dW': dW, 'db': db}
    return grads, cost


# In[304]:


def optimize(W, b, X, Y, num_itr=1000, alpha=0.05, print_cost=False):
    costs = []
    
    for i in range(num_itr):
        grads, cost = Prop(X, W, b, Y, i)
        
        dW, db = grads['dW'], grads['db']
        W = W - np.dot(alpha, dW)
        b = b - alpha*db
        
        if i%100 == 0:
            costs.append(cost)
        
        if print_cost and i%100 == 0:
            print('cost at itr:', i, 'is:', cost)
    
    grads['dW'] = dW
    grads['db'] = db
    
    params = {'W': W, 'b': b}
    
    return params, grads, cost


# In[305]:


def predict(W, b, X):
    predictY = np.zeros((1, X.shape[1]))
    
    Z = np.dot(W.T, X) + b
    A = sigmoid(Z)
    
    for i in range(A.shape[1]):
        if A[0][i] > 0.5:
            predictY[0][i] = 1
        else:
            predictY[0][i] = 0

    return predictY


# In[340]:


def model(X_train, Y_train, X_test, Y_test, num_itr=1000, alpha=0.01, print_cost=False):
    n_x, m = X_train.shape[0], X_train.shape[1]
    
    W, b = initialiseParams(n_x)
    
    prams, grads, costs = optimize(W, b, X_train, Y_train, num_itr, alpha, True)
    
    W = prams['W']
    b = prams['b']
    
    Y_predict_test = predict(W, b, X_test)
    Y_predict_train = predict(W, b, X_train)
    
    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_predict_train - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_predict_test - Y_test)) * 100))
    
    d = {"costs": costs,
         "Y_prediction_test": Y_predict_test, 
         "Y_prediction_train" : Y_predict_train, 
         "w" : W, 
         "b" : b,
         "learning_rate" : alpha,
         "num_iterations": num_itr}
    return d


# In[341]:


d = model(train_x_flatten, train_y, test_x_flatten, test_y)


# In[337]:


y_predict = d['Y_prediction_test']


# In[330]:


for i in range(test_x_orig.shape[0]):
    print('i:', i, 'predicted:', int(y_predict[0][i]), 'vs is cat:', test_y[0][i])
    plt.imshow(test_x_orig[i])
    plt.show()
    print('------------------------------')

