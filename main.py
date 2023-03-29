from neuralNetwork import *
from pic2list import *

#parameters:
inputNodes = 3
hiddenNodes = 3
outputNodes = 3
learningRate = 0.3

n = neuralNetwork(inputNodes, hiddenNodes, outputNodes, learningRate)

a = n.query([1.0, 0.5, -1.5])
print(a)

#Open pictures

#reshape til 32*32 billede (evt. biblioteket pillow) og tag pixelværdier og sæt i rækkefølge efter 
#hinanden (numpy.reshape) således du får en streng adskilt af komma'er. Skal billedet croppes eller 
#sætte black bars på?

#n.debug()