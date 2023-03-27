import numpy #used for matrixes
import scipy.special #used for sigmoid-curve

class neuralNetwork:

    #initialise network
    def __init__(self, inputNodes,hiddenNodes, outputNodes, learningRate):       
        #set numbers of nodes for each layer
        self.inputNodes = inputNodes
        self.hiddenNodes = hiddenNodes
        self.outputNodes = outputNodes

        #set learning rate 
        self.learningRate = learningRate

        #weights
        self.w_input_hidden = (numpy.random.rand(self.hiddenNodes, self.inputNodes) - 0.5)
        self.w_hidden_output = (numpy.random.rand(self.outputNodes, self.hiddenNodes) - 0.5)

        #function
        self.activationFunction = lambda x: scipy.special.expit(x)

        pass

#    def activationFunction(x):
#            return scipy.special.expit(x)

    #train function
    def train(self, inputsList, targetsList):
        #--Calculate the output (just as query)--#

        #converts input to 2D array
        inputs = numpy.array(inputsList, ndmin=2).T

        #calculate hidden layer
        hiddenInputs = numpy.dot(self.w_input_hidden, inputs)
        hiddenOutputs = self.activationFunction(hiddenInputs)

        #calculate output layer
        finalInputs = numpy.dot(self.w_hidden_output, hiddenOutputs)
        finalOutputs = self.activationFunction(finalInputs)

        #--Training on the output--#
        
        #converts to 2D array
        targets = numpy.array(targetsList, ndmin=2).T

        #calculate output error
        outputErrors = targets - finalOutputs

        #hidden layer error
        hiddenErrors = numpy.dot(self.w_hidden_output.T, outputErrors)

        #update the weights (first input to hidden weights, then hidden to output weights)
        self.w_input_hidden  += self.learningRate * numpy.dot((hiddenErrors * hiddenOutputs * (1.0 - hiddenOutputs)), 
                                                              numpy.transpose(hiddenOutputs))
        self.w_hidden_output += self.learningRate * numpy.dot((outputErrors * finalOutputs * (1.0 - finalOutputs)), 
                                                              numpy.transpose(inputs))
        
        pass

    #query/answer
    def query(self, inputsList):
        #converts input to 2D array
        inputs = numpy.array(inputsList, ndmin=2).T

        #calculate hidden layer
        hiddenInputs = numpy.dot(self.w_input_hidden, inputs)
        hiddenOutputs = self.activationFunction(hiddenInputs)

        #calculate output layer
        finalInputs = numpy.dot(self.w_hidden_output, hiddenOutputs)
        finalOutputs = self.activationFunction(finalInputs)

        return finalOutputs

    def debug(self):
        print("input->hidden:\n", self.w_input_hidden)
