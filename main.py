from neuralNetwork import *
from pic2list import *
import dill
import random

def resize_img_forloop(length, img_path, image_list, answer_list, answer, x):
    for i in range(length):
        img_path_img = img_path + f".{i+x}.jpg"
        pixels = resize_image(img_path_img)

        image_list.append(pixels)
        answer_list.append(answer)

        print(i)

    return image_list, answer_list

#Dog = 0, Cat = 1

#parameters:
inputNodes = 1024
hiddenNodes = 1024
outputNodes = 1
learningRate = 0.3

n = neuralNetworkBook(inputNodes, hiddenNodes, outputNodes, learningRate)

pickle_answer = input("Do you want to train again? y/n")
if pickle_answer == "y":

    n.debug()

    pickle_resize_answer_test = input("resize images again? y/n")
    if pickle_resize_answer_test == "y":
        #Open pictures
        trainList = []
        trainAnswerList = []

        img_path_0 = "C:/Users/jacob/OneDrive - Syddansk Erhvervsskole/Programering/Eksamensprojekt/dataset/training_set/cats/cat"
        trainList, trainAnswerList = resize_img_forloop(4000, img_path_0, trainList, trainAnswerList, 1, 1)

        img_path_1 = f"C:/Users/jacob/OneDrive - Syddansk Erhvervsskole/Programering/Eksamensprojekt/dataset/training_set/dogs/dog"
        trainList, trainAnswerList = resize_img_forloop(4000, img_path_1, trainList, trainAnswerList, 0, 1)
        
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


##TEST##
correct_answers = 0
wrong_answers = 0

pickle_test_answer = input("resize image again? y/n")
if pickle_test_answer == "y":
    testList = []
    testAnswerList = []

    img_path_2 = "C:/Users/jacob/OneDrive - Syddansk Erhvervsskole/Programering/Eksamensprojekt/dataset/test_set/cats/cat"
    testList, testAnswerList = resize_img_forloop(1000, img_path_2, testList, testAnswerList, 1, 4001)

    img_path_3 = "C:/Users/jacob/OneDrive - Syddansk Erhvervsskole/Programering/Eksamensprojekt/dataset/test_set/dogs/dog"
    testList, testAnswerList = resize_img_forloop(1000, img_path_3, testList, testAnswerList, 0, 4001)
    
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
    assumed_answer = int(n.query(testList[i]))
    correct_answer = testAnswerList[i]

    if assumed_answer == correct_answer:
        correct_answers += 1
    else:
        wrong_answers += 1
        print(correct_answer)
    
    print(f't:{i}')

n.debug()

print(f'correct answers = {correct_answers}')
print(f'wrong answers = {wrong_answers}')

input('press enter to close')