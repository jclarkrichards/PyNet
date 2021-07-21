from network import Network
import numpy as np

net = Network()
net.setInput([2,3,5], 10)
net.setExpected([10,4,0], 10)
#net.addLayer(4)
net.addLayer(10)
#net.addLayer(10)
#net.addLayer(10)
net.addLayer(3)

print("\nLayers")
for layer in net.layers:
    print(layer)
    print("")

    
net.setWeights()
print("\nInitial Weights")
for w in net.weights:
    print(w)
    print("")

print("\n\nForward Pass")
y = net.forward()
print("Layers")
for layer in net.layers:
    print(layer)
print("\nResults")
print(y)
print("\nExpected")
print(net.expected)


error = y - net.expected
print("\nError")
print(error)


#print("\n\nBack Propogation")
#net.back(y)

#print("\n\nForward again")
#y = net.forward()
#print("\nResults")
#print(y)
#print("\nExpected")
#print(net.expected)

#error = y - net.expected
#print("\nError")
#print(error)


for i in range(100):
    print("Interation " + str(i))
    y = net.forward()
    net.back(y)


print("\n\n\nFinal Results")
print(y)
print("\nExpected")
print(net.expected)

error = y - net.expected
print("\nError")
print(error)
