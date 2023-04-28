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
        self.w_input_hidden = numpy.random.normal(0.0, pow(self.inputNodes, -0.5), (self.hiddenNodes, self.inputNodes))
        self.w_hidden_output = numpy.random.normal(0.0, pow(self.hiddenNodes, -0.5), (self.outputNodes, self.hiddenNodes))

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
        self.w_hidden_output += self.learningRate * numpy.dot((outputErrors * finalOutputs * (1.0 - finalOutputs)), 
                                                              numpy.transpose(hiddenOutputs))
        self.w_input_hidden  += self.learningRate * numpy.dot((hiddenErrors * hiddenOutputs * (1.0 - hiddenOutputs)), 
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
        print(self.w_input_hidden)
        print(self.w_hidden_output)



# neural network class definition
class neuralNetworkBook:
    
    # initialise the neural network
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        # set number of nodes in each input, hidden, output layer
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        
        # link weight matrices, wih and who
        # weights inside the arrays are w_i_j, where link is from node i to node j in the next layer
        # w11 w21
        # w12 w22 etc 
        self.wih = numpy.random.normal(0.0, pow(self.inodes, -0.5), (self.hnodes, self.inodes))
        self.who = numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.onodes, self.hnodes))

        # learning rate
        self.lr = learningrate
        
        # activation function is the sigmoid function
        self.activation_function = lambda x: scipy.special.expit(x)
        
        pass

    
    # train the neural network
    def train(self, inputs_list, targets_list):
        # convert inputs list to 2d array
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T
        
        # calculate signals into hidden layer
        hidden_inputs = numpy.dot(self.wih, inputs)
        # calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)
        
        # calculate signals into final output layer
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)
        
        # output layer error is the (target - actual)
        output_errors = targets - final_outputs
        # hidden layer error is the output_errors, split by weights, recombined at hidden nodes
        hidden_errors = numpy.dot(self.who.T, output_errors) 
        
        # update the weights for the links between the hidden and output layers
        self.who += self.lr * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)), numpy.transpose(hidden_outputs))
        
        # update the weights for the links between the input and hidden layers
        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), numpy.transpose(inputs))
        
        pass

    
    # query the neural network
    def query(self, inputs_list):
        # convert inputs list to 2d array
        inputs = numpy.array(inputs_list, ndmin=2).T
        
        # calculate signals into hidden layer
        hidden_inputs = numpy.dot(self.wih, inputs)
        # calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)
        
        # calculate signals into final output layer
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)
        
        return final_outputs
    
    def debug(self):
        print(self.wih)
        print(self.who)