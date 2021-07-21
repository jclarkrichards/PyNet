from network import Network
import numpy as np
from functions import *

net = Network()
net.setInput([.9, .1, .8])
net.addLayer(3)
net.addLayer(10)
net.setExpected([.2, .6, .3])

net.setWeights()
#W0 = np.array([[.9, .3, .4], [.2, .8, .2], [.1, .5, .6]])
#W1 = np.array([[.3, .7, .5], [.6, .5, .2], [.8, .1, .9]])
#net.addWeight(W0)
#net.addWeight(W1)
net.setActivation(leaky_relu)

print("Layers")
for layer in net.layers:
    print(layer)
    print("\n")


print("\n\nWeights")
for weight in net.weights:
    print(weight)
    print("\n")


#print("W0 x I")
#x = mat_vec(net.weights[0], net.layers[0])
#print(x)

#print("\nPass through activation function")
#x = net.activate(x)
#print(x)

#print("\nPass through to next layer")
#x = mat_vec(net.weights[1], x)
#print(x)
#print("\nPass through activation function")
#x = net.activate(x)
#print(x)


#print("\nDo again in one fell swoop")
#net.forward()
#print("Layers")
#for layer in net.layers:
#    print(layer)
#    print("\n")


#print("Outputs")
#print(net.outputs)


#print("\nLets find the error vectors")
#print(net.errorProp())
#print("\nBack Propogation")
#net.backProp()
#for weight in net.weights:
#    print(weight)
#    print("\n")

#print("\nForward again")
#net.forward()
#print(net.outputs)
print("\n\nResults")
for i in range(300):
    net.forward()
    net.backProp()

print(net.expected)
print("\n")
print(net.outputs)




