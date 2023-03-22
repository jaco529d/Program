from neuralNetwork import *

#parameters:
inputNodes = 3
hiddenNodes = 3
outputNodes = 3
learningRate = 0.3

n = neuralNetwork(inputNodes, hiddenNodes, outputNodes, learningRate)

n.debug()