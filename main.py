from neuralNetwork import *

#parameters:
inputNodes = 3
hiddenNodes = 3
outputNodes = 3
learningRate = 0.3

n = neuralNetwork(inputNodes, hiddenNodes, outputNodes, learningRate)

a = n.query([1.0, 0.5, -1.5])
print(a)
#n.debug()