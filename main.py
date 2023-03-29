from neuralNetwork import *
from pic2list import *

#parameters:
inputNodes = 1024
hiddenNodes = 1024
outputNodes = 1
learningRate = 0.3

n = neuralNetwork(inputNodes, hiddenNodes, outputNodes, learningRate)

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

for i in range(len(trainList)):
    n.train(trainList[i], trainAnswerList[i])

catTest = n.query(resize_image('C:/Users/jacob/OneDrive - Syddansk Erhvervsskole/Programering/Eksamensprojekt/dataset/test_set/cats/cat.4001.jpg'))
dogTest = n.query(resize_image('C:/Users/jacob/OneDrive - Syddansk Erhvervsskole/Programering/Eksamensprojekt/dataset/test_set/dogs/dog.4001.jpg'))

print("cat: ", int(catTest))
print("dog: ", int(dogTest))
input('press enter to close')

#reshape til 32*32 billede (evt. biblioteket pillow) og tag pixelværdier og sæt i rækkefølge efter 
#hinanden (numpy.reshape) således du får en streng adskilt af komma'er. Skal billedet croppes eller 
#sætte black bars på?

#n.debug()