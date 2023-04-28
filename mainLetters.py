from neuralNetwork import *
from pic2list import *
import dill
import random

def resize_img_forloop(length, img_path, image_list, answer_list, answer, x):
    for i in range(length):
        img_path_img = img_path + f".{i+x}.jpg"
        pixels = resize_image(img_path_img)

        image_list.append(pixels)
        if answer == 1:
             answer_list.append([1,0])
        else:
             answer_list.append([0,1])

        print(i)

    return image_list, answer_list

#Dog = 0, Cat = 1

#parameters:
inputNodes = 784
hiddenNodes = 784
hidden2Nodes = 200
outputNodes = 10
learningRate = 0.1

n = neuralNetwork(inputNodes, hiddenNodes, outputNodes, learningRate)

pickle_answer = input("Do you want to train again? y/n")
if pickle_answer == "y":

    n.debug()

    training_data_file = open("C:/Users/jacob/OneDrive - Syddansk Erhvervsskole/Programering/Eksamensprojekt/dataset/mnist_train.csv", 'r')
    training_data_list = training_data_file.readlines()
    training_data_file.close()

    for record in training_data_list:
        # split the record by the ',' commas
        all_values = record.split(',')
        # scale and shift the inputs
        inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        # create the target output values (all 0.01, except the desired label which is 0.99)
        targets = numpy.zeros(outputNodes) + 0.01
        # all_values[0] is the target label for this record
        targets[int(all_values[0])] = 0.99
        n.train(inputs, targets)
        pass
    
    use_pickle_answer = input("do you want to pickel this neural network? y/n")
    if use_pickle_answer == "y":
        with open("neuralNetwork.pickle", "wb") as f:
            dill.dump(n, f)
    else:
         pass

elif pickle_answer == "n":
    with open("neuralNetwork.pickle", "rb") as f:
        n = dill.load(f)
    print("continue with pickeled, now testing")


##TEST##
# load the mnist test data CSV file into a list
test_data_file = open("C:/Users/jacob/OneDrive - Syddansk Erhvervsskole/Programering/Eksamensprojekt/dataset/mnist_test.csv", 'r')
test_data_list = test_data_file.readlines()
test_data_file.close()

# test the neural network

# scorecard for how well the network performs, initially empty
scorecard = []

# go through all the records in the test data set
for record in test_data_list:
    # split the record by the ',' commas
    all_values = record.split(',')
    # correct answer is first value
    correct_label = int(all_values[0])
    # scale and shift the inputs
    inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
    # query the network
    outputs = n.query(inputs)
    # the index of the highest value corresponds to the label
    label = numpy.argmax(outputs)
    # append correct or incorrect to list
    if (label == correct_label):
        # network's answer matches correct answer, add 1 to scorecard
        scorecard.append(1)
    else:
        # network's answer doesn't match correct answer, add 0 to scorecard
        scorecard.append(0)
        pass
    
    pass
# calculate the performance score, the fraction of correct answers
scorecard_array = numpy.asarray(scorecard)
print ("performance = ", scorecard_array.sum() / scorecard_array.size)


n.debug()

input('press enter to close')