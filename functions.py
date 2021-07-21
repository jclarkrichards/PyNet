from typing import Callable
import numpy as np
from numpy import ndarray

def deriv_call(func: Callable[[ndarray], ndarray], input_: ndarray, delta: float=0.0001) -> ndarray:
    return (func(input_ + delta) - func(input_ - delta)) / (2 * delta)

def deriv(func, input_: ndarray, delta: float=0.0001) -> ndarray:
    '''Returns an approximation of the derivative of the function at given points'''
    #return (func(input_ + delta) - func(input_)) / delta
    return (func(input_ + delta) - func(input_ - delta)) / (2 * delta)

def deriv_chain(funcs: ndarray, input_: ndarray, delta: float=0.0001) -> ndarray:
    '''Return approximate derivative of a nest of functions.  (Chain rule).  Innermost function first'''
    f1x = funcs[0](input_)
    temp = [f1x] #will have len(funcs) - 1 entries
    for i in list(range(1, len(funcs)-1)):
        temp.append(funcs[i](temp[i-1]))

    result = deriv(funcs[0], input_)
    for i in list(range(len(temp))):
        result *= deriv(funcs[i+1], temp[i])
    return result

def deriv_backprop(funcs: ndarray, input1_: ndarray, intput2_: ndarray, delta: float=0.0001) -> ndarray:
    '''Return approximate derivative of a nest of functions.  (Chain rule).  Innermost function first.  Each function requires 2 inputs.'''
    f1x = funcs[0](input1_, input2_)
    temp = [f1x] #will have len(funcs) - 1 entries
    for i in list(range(1, len(funcs)-1)):
        temp.append(funcs[i](temp[i-1]))

    result = deriv(funcs[0], input_)
    for i in list(range(len(temp))):
        result *= deriv(funcs[i+1], temp[i])
    return result

def func_chain(funcs: ndarray, input_: ndarray, delta: float=0.0001) -> ndarray:
    '''Returns result of nested functions [f0, f1, f2, ..., fn] such that fn(...f2(f1(f0(input_))))'''
    temp = funcs[0](input_)
    for i in list(range(1, len(funcs))):
        temp = funcs[i](temp)
    return temp

"""========="""
"""Functions"""
"""========="""
def square(x: ndarray) -> ndarray:
    return np.power(x, 2)

def leaky_relu(x: ndarray) -> ndarray:
    return np.maximum(0.2 * x, x)

def relu(x: ndarray) -> ndarray:
    return np.maximum(0, x)

def polytest(x:ndarray) -> ndarray:
    return np.power(x, 3) + 2*np.power(x, 2) - 3*x + 10

def sigmoid(x: ndarray) -> ndarray:
    return 1 / (1 + np.exp(-x))

def sigmoidPrime(x: ndarray) -> ndarray:
    return sigmoid(x) * (1 - sigmoid(x))

def error(t: ndarray, x: ndarray) -> ndarray:
    '''Defines the error function given the expected (t) and actual (x)'''
    return np.power(t - x, 2)

def mat_vec(M: ndarray, x: ndarray) -> ndarray:
    '''Function that multiplies a Matrix and a vector together'''
    return np.dot(M, x)
