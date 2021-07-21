import numpy as np
from numpy import ndarray
from functions import *

"""
Defines a Neural Network with layers, weights, activation functions, etc.
Any number of layers can be added and each layer can be any size.

A numpy array is in the form of (rows, columns)
"""

class Network(object):
    def __init__(self):
        self.inputs = None  #Inputs (doesn't change)
        self.outputs = None #The final output of current forward pass
        self.expected = None #a list of values we expect to get
        self.weights = [] #List of weights, depends on the number of layers
        self.layers = [] #Each layer is a vector.  Contains the output of layer
        self.alpha = 0.01 #Learning rate
        self.sigma = None #Activation function

    def addLayer(self, size):
        '''Add a layer of any size.  Layer has shape (size, 1)'''
        self.layers.append(np.zeros((size,1)))
        
    def setWeights(self):
        '''Sets initial weights depending on the number of layers'''
        for i in list(range(len(self.layers)-1)):
            size = (self.layers[i+1].shape[0], self.layers[i].shape[0])
            #W = np.zeros(size)
            W = np.random.normal(0.01, 1, size)
            self.weights.append(W)

    def addWeight(self, weights):
        '''For testing or if need to add weights by hand'''
        self.weights.append(weights)
    """    
    def relu(self, n):
        '''Modified ReLU where negative values remain but small'''
        if n < 0:
            return n * 0.02
        return n

    def reluPrime(self, n):
        '''Derivative of the Modified ReLU'''
        if n <= 0:
            return 0.02
        return 1

    def sigmoid(self, x: ndarray) -> ndarray:
        '''Sigmoid function can be an activation function'''
        return 1 / (1 + np.exp(-x))
    """
    def setActivation(self, func):
        '''sigma is the activation function'''
        self.sigma = func
    """
    def activate(self, x: ndarray) -> ndarray:
        '''Pass data through the activation function'''
        return self.sigma(x)
    
    def activation(self, vector):
        '''This is the activation function. Must input a vector of data'''
        x = map(self.relu, list(vector.flatten()))
        return np.array(list(x)).reshape((len(vector), 1))
    
    def activationPrime(self, vector):
        '''Get the derivative of the activation function'''
        x = map(self.reluPrime, list(vector.flatten()))
        return np.array(list(x)).reshape((len(vector), 1))        
    """
    def setInput(self, inputlist, maxval=None):
        '''Get a list of inputs, vectorize and normalize it
        Input list can be tuple or list of values'''
        self.inputs = np.array(inputlist).reshape((len(inputlist),1))
        if maxval is not None:
            self.inputs = self.normalize(self.inputs, maxval)
        self.layers.insert(0, self.inputs) #Input is always the first layer

    def setExpected(self, inputlist, maxval=None):
        self.expected = np.array(inputlist).reshape((len(inputlist),1))
        if maxval is not None:
            self.expected = self.normalize(self.expected, maxval)
        self.layers.insert(len(self.layers), np.zeros_like(self.expected))
        
    """    
    def normalize(self, npvec, maxval):
        '''Normalize the np vector and return the result'''
        return (npvec / maxval * 0.99) + 0.01
    
    def matrixMult(self, W, I):
        '''Matrix multiply WxI and return the result'''
        return np.dot(W, I)
    """
    def forward(self):
        '''Perform the forward pass.  End with the final computed values'''
        #print("FORWARD PASS")
        for i in list(range(len(self.weights))):
            x = mat_vec(self.weights[i], self.layers[i])
            self.layers[i+1] = self.sigma(x)
        self.outputs = self.layers[-1]
        print(self.outputs)
        
    def errorProp(self):
        '''Find all of the errors for each Weights'''
        #print("=========ERROR CALC")
        errors = [0] * len(self.weights)
        #print(errors)
        errors[-1] = self.expected - self.outputs
        #print(errors)
        for i in list(range(len(self.weights)-2, -1, -1)):
            #print(i)
            #print(self.weights[i+1])
            #print(errors[i+1])
            errors[i] = np.dot(self.weights[i+1].T, errors[i+1])
        return errors


    def backProp(self):
        errors = self.errorProp()
        for i in list(range(len(self.weights))):
            dW = -errors[i] * deriv(self.sigma, np.dot(self.weights[i], self.layers[i])) * self.layers[i].T
            self.weights[i] = self.weights[i] - dW * self.alpha
        
    """
    def back(self, output):
        '''Perform back propogation.  This updates the weights.'''
        print("===========================================")
        print("==========BACK PROPOGATION START===========")
        print("===========================================")
        for i in list(range(len(self.weights)-1, -1, -1)):
            print("\n\n" + str(i) + "  ==================== ")
            print("\nWeight["+str(i)+"]")
            print(self.weights[i])
            print("\nx["+str(i+1)+"]")
            print(self.layers[i+1])
            if i == len(self.weights)-1:
                E = self.layers[i+1] - self.expected
            else:
                E = -np.dot(self.weights[i+1].T, self.layers[i+2])

            print("\nError["+str(i+1)+"]")
            print(E)
            junk = np.dot(self.weights[i], self.layers[i])
            prime = self.activationPrime(junk)
            print("\nW[i] . x[i+1]")
            print(prime)
            temp = E * prime
            print("\nPrime")
            print(temp)
            dW = np.dot(temp, self.layers[i].T)
            print("\ndW")
            print(dW)
            self.weights[i] = self.weights[i] - dW*self.alpha
    """
