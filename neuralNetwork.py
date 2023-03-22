import numpy

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

        pass

    #train function
    def train(self):
        pass

    #query/answer
    def query(self):
        pass

    def debug(self):
        print("input->hidden:\n", self.w_input_hidden)
