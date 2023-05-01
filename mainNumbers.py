from neuralNetwork import *
from pic2list import *
import dill
import random

#parameters:
inputNodes = 784
hiddenNodes = 300
outputNodes = 10
learningRate = 0.1

n = neuralNetwork(inputNodes, hiddenNodes, outputNodes, learningRate)

pickle_answer = input("Do you want to train again? y/n")
if pickle_answer == "y":

    allTrainNumbers = createNumbersList(10000)
    print(f'generated {len(allTrainNumbers)} numbers')

    for k in range(5):
        for all_values in allTrainNumbers:
            #print(type(all_values), all_values)
            inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
            # create the target output values (all 0.01, except the desired label which is 0.99)
            targets = numpy.zeros(outputNodes) + 0.01
            # all_values[0] is the target label for this record
            targets[int(all_values[0])] = 0.99
            n.train(inputs, targets)
            pass
        print(k)
        pass
    
    use_pickle_answer = input("do you want to pickel this neural network? y/n")
    if use_pickle_answer == "y":
        with open("neuralNetwork.pickle", "wb") as f:
            dill.dump(n, f)
    else:
         pass

else:
    with open("neuralNetwork.pickle", "rb") as f:
        n = dill.load(f)
    print("continue with pickeled, now testing")


##TEST##
correct_answers = 0
unknown = 0
wrong_answers = 0

allTestNumbers = createNumbersList(10000)
print(f'generated {len(allTestNumbers)} numbers')

for all_values in allTestNumbers:
    correct_label = int(all_values[0])
    inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
    # create the target output values (all 0.01, except the desired label which is 0.99)
    outputs = n.query(inputs)
    label = numpy.argmax(outputs)
    
    if label == correct_label:
        correct_answers += 1
    else:
        wrong_answers += 1
        


print(f'correct answers = {correct_answers}')
print(f'wrong answers = {wrong_answers}')
#print(f'unknown answers = {unknown}')

input('press enter to close')