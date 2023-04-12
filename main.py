from neuralNetwork import *
from pic2list import *
import dill
import random

#parameters:
inputNodes = 1024
hiddenNodes = 1024
outputNodes = 1
learningRate = 0.3

n = neuralNetwork(inputNodes, hiddenNodes, outputNodes, learningRate)

pickle_answer = input("Do you want to train again? y/n")
if pickle_answer == "y":

    pickle_resize_answer_test = input("resize images again? y/n")
    if pickle_resize_answer_test == "y":
        #Open pictures
        trainList = []
        trainAnswerList = []

        for i in range(4000):
            i += 1

            img_path = f"C:/Users/jacob/OneDrive - Syddansk Erhvervsskole/Programering/Eksamensprojekt/dataset/training_set/cats/cat.{i}.jpg"
            pixels = resize_image(img_path)
            
            trainList.append(pixels)
            trainAnswerList.append(1)
            print(i)

        for i in range(4000):
            i += 1

            img_path = f"C:/Users/jacob/OneDrive - Syddansk Erhvervsskole/Programering/Eksamensprojekt/dataset/training_set/dogs/dog.{i}.jpg"
            pixels = resize_image(img_path)
            
            trainList.append(pixels)
            trainAnswerList.append(0)
            print(i)
        
        with open("train_img.pickle", "wb") as f:
            dill.dump(trainList, f)
        with open("train_img_answer.pickle", "wb") as f:
            dill.dump(trainAnswerList, f)

    else:
        with open("train_img.pickle", "rb") as f:
            trainList = dill.load(f)
    
        with open("train_img_answer.pickle", "rb") as f:
            trainAnswerList = dill.load(f)

    for i in range(len(trainList)):
        k = random.randint(0 , (len(trainList) - 1))
        n.train(trainList[k], trainAnswerList[k])

        trainList.pop(k)
        trainAnswerList.pop(k)

        print(i)
    
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

#catTest = n.query(resize_image('C:/Users/jacob/OneDrive - Syddansk Erhvervsskole/Programering/Eksamensprojekt/dataset/test_set/cats/cat.4001.jpg'))
#dogTest = n.query(resize_image('C:/Users/jacob/OneDrive - Syddansk Erhvervsskole/Programering/Eksamensprojekt/dataset/test_set/dogs/dog.4001.jpg'))

##TEST##
correct_answers = 0
wrong_answers = 0

pickle_test_answer = input("resize image again? y/n")
if pickle_test_answer == "y":
    testList = []
    testAnswerList = []

    for i in range(1000):
        i += 1

        img_path = f"C:/Users/jacob/OneDrive - Syddansk Erhvervsskole/Programering/Eksamensprojekt/dataset/test_set/cats/cat.{i+4000}.jpg"
        pixels = resize_image(img_path)

        testList.append(pixels)
        testAnswerList.append(1)
        print(f'r:{i}')

    for i in range(1000):
        i += 1

        img_path = f"C:/Users/jacob/OneDrive - Syddansk Erhvervsskole/Programering/Eksamensprojekt/dataset/test_set/dogs/dog.{i+4000}.jpg"
        pixels = resize_image(img_path)

        testList.append(pixels)
        testAnswerList.append(0)
        print(f'r:{i}')
    
    with open("test_img.pickle", "wb") as f:
            dill.dump(testList, f)
    with open("test_img_answer.pickle", "wb") as f:
            dill.dump(testAnswerList, f)

else:
    with open("test_img.pickle", "rb") as f:
            testList = dill.load(f)
    
    with open("test_img_answer.pickle", "rb") as f:
            testAnswerList = dill.load(f)

for i in range(len(testList)):
    aa = n.query(testList[i])
    assumed_answer = int(aa)
    correct_answer = testAnswerList[i]

    if assumed_answer == correct_answer:
        correct_answers += 1
    else:
        wrong_answers += 1
    
    print(f't:{i}')

print(f'correct answers = {correct_answers}')
print(f'wrong answers = {wrong_answers}')

#print("cat: ", int(catTest))
#print("dog: ", int(dogTest))


input('press enter to close')

#reshape til 32*32 billede (evt. biblioteket pillow) og tag pixelværdier og sæt i rækkefølge efter 
#hinanden (numpy.reshape) således du får en streng adskilt af komma'er. Skal billedet croppes eller 
#sætte black bars på?

#n.debug()