import numpy as np
import pandas as pd
import sys, math
import matplotlib.pyplot as plt
import matplotlib.image as matimg

np.set_printoptions(suppress = True)
np.set_printoptions(threshold = sys.maxsize)

#testIMG = open('/Users/MNIST Data/train-images-idx3-ubyte', 'rb')
#testLB = open('/Users/MNIST Data/train-labels.idx1-ubyte', 'rb')

pixelArr = np.array([])
#pixelArr = pixelArr[np.newaxis,:]
pixelPrt = np.array([])

labelArr = np.array([])
#labelArr = labelArr[np.newaxis,:]
labelPrt = np.array([])

MASTER_num_of_pic = 10

#____________________________________________________________
#These for loops store non image bytes into arrays
trainingINFO = np.array([])
trainingINFOlabel = np.array([])
def STEP1(testIMG1, testLB1):
    #testIMG = open('/Users/MNIST Data/train-images-idx3-ubyte', 'rb')
    #testLB = open('/Users/MNIST Data/train-labels.idx1-ubyte', 'rb')
    for z in range(4):
        first_str = testIMG1.read(4)
        #NOTE: read reads "(size)"bytes  at a time, in this case 4
        #print(type(str(first_str)), "   ", first_str, int.from_bytes(first_str, byteorder = 'big'))
        
        np.append(trainingINFO, int.from_bytes(first_str, byteorder = 'big'))
        #NOTE: from_bytes converts bytes into integers, byteorder = 'big' means read from right to left
    for z in range(2):
        first_str = testLB1.read(4)
        np.append(trainingINFOlabel, int.from_bytes(first_str, byteorder = 'big'))
            #print(first_str, "  ",  int.from_bytes(first_str, byteorder = 'big'))
    print("pixelArr.shape", pixelArr.shape, pixelArr)
#--------------------------------------------------------------------------------

#______________________________________________________________
# Initiat pixelArr to shape and size for more arrays to be added
def STEP2(num_of_pic, testIMG, testLB):
    for z in range(28*28):
        global pixelArr
        first_str = testIMG.read(1)
        pixelArr = np.append(pixelArr, int.from_bytes(first_str, byteorder = 'big'))
        #no idea why pixelArr needs to equal itself in this line ^^^^ 
    pixelArr = pixelArr[np.newaxis,:]
    print(pixelArr.shape, " pixelArr results", pixelArr)
    #Initiates array of labels in correct order
    # +1 because we already put one picture in before^^^^^
    for z in range(num_of_pic + 1):
        global labelArr
        first_str2 = testLB.read(1)
        labelArr =  np.append(labelArr, int.from_bytes(first_str2, byteorder = 'big'))

#print(labelArr)
#--------------------------------------------------------------------------------

#______________________________________________________________
# Converts images to use, controls how many images in pixelArr
#takes images and stores it in pixelArr in a series of 1-D arrays
def STEP3(num_of_pic, testIMG, testLB):
    for i in range(num_of_pic):
        global pixelArr, labelPrt
        for z in range(28*28):
            global pixelPrt
            first_str = testIMG.read(1)
            pixelPrt = np.append(pixelPrt, int.from_bytes(first_str, byteorder = 'big'))
            
        pixelPrt = pixelPrt[np.newaxis,:]
        pixelArr = np.append(pixelArr, pixelPrt, axis = 0)
        #Some reason, axis need to be set to zero to prevent colapsing to 1-D
        pixelPrt = []

    testIMG.close()
    testLB.close()

    labelPrt = labelPrt[:,np.newaxis]

    print( '--------')
    print("PixelArr: ", pixelArr.shape)
    print("labelArr: ", labelArr,labelArr.shape)
    print("labelPrt: ", labelPrt, labelPrt.shape)
#pixelArr = pixelArr.reshape((MASTER_num_of_pic + 1, 28, 28))
#^^^ required for displaying image in next section ln 107
#print(pixelArr.shape, pixelArr)

#--------------------------------------------------------------------------------

#______________________________________________________________
#turns label arrays into more arrays in [000...1...00] sort of format
emptyArr = np.array([0,0,0,0,0,0,0,0,0])
labelArr2 = np.array([0,0,0,0,0,0,0,0])
labelPrt2 = np.array([])

#initial set up for array structure
def STEP4(testIMG, testLB):
    global labelArr2, emptyArr
    labelArr2 = np.insert(emptyArr,int(labelArr[0]),1)
    labelArr2 = labelArr2[np.newaxis,:]
    for i in range(1,labelArr.size):
        labelPrt2 = np.insert(emptyArr,int(labelArr[i]),1)
        labelPrt2 = labelPrt2[np.newaxis,:]
        labelArr2 = np.append(labelArr2,labelPrt2,axis = 0)
        labelPrt2 =[]
    print("Converted to label table, labelArr2:\n", labelArr2,"\n Actual numbers, label Arr:\n", labelArr)
    
    #plt.imshow(np.reshape(pixelArr[0], (28,28)), interpolation = 'none')
    #plt.show()
    #^^^ Shows the image!! Sucesss 
    
    print("start up complete")
#--------------------------------------------------------------------------------

#______________________________________________________________
#Initialization of weights and biases which also act as the "layers"
# 16 is the the number of nodes in a  layer, such as layer 1, 0,
#the starting layer 0 is size 764, the number of pixels
#last layer has 10 nodes, to represent 10 possible outcomes

L1weights = 4*np.random.random_sample((16,28*28)) -2
L1biases = 4*np.random.random_sample((16)) -2

L2weights = 4*np.random.random_sample((16,16))-2
L2biases = 4*np.random.random_sample((16))-2

L3weights = 4*np.random.random_sample((10,16)) -2
L3biases = 4*np.random.random_sample((10)) -2
#print("--------", "(Section on weights)" )
#print(L1weights.shape, " ", L2weights.shape, " ", L3weights.shape)
#print("--------", "(Section on Biases)" )
#print(L1biases.shape, " ", L2biases.shape, " ", L3biases.shape)
#Function to make a new array of random numbers:
def randWeight16x764Gen():
    return (4*np.random.random_sample((16,28*28))-2)
def randWeight10x16Gen():
    return (4*np.random.random_sample((10,16)) -2)
def randWeight16x16Gen():
    return (4*np.random.random_sample((16,16)) -2)

def randBiases16Gen():
    return (4*np.random.random_sample((16))-2)
def randBiases10Gen():
    return (4*np.random.random_sample((10)) -2)

#------------------------------------------------------------------------------

#______________________________________________________________
#Function implimentation of relu, also allows you "map" the function across the whole vector.
def reluF(x):
    return np.maximum(0, x)

#Defining a function that takes in weights of shape (16, 784) and image of shape (784)
#multiplied to get output vector of shape (16), doesnt actually control shape

#The image = the Nodes
def matrixMultBias(weight, biases,  image, index = None):
    if(index == None):
        #breakpoint()
        return reluF(np.matmul(weight, image) + biases)
    else:
        return reluF(np.matmul(weight, image[index]) + biases)

def FunctionTesting():
    print("-------(Function Testing)")
    print("pixelArr.shape: ", pixelArr.shape)
    print("matmul of randGen and pixelArr: ", np.matmul(randWeight16x764Gen(), pixelArr[0]).shape)
    print("matrixMultBias test: ", matrixMultBias(randWeight16x764Gen(), L1biases, pixelArr, 0))
    #print(matrixMult(L1weights, L1biases, pixelArr, 0))
    print("randWeight16x764Gen shape: ", randWeight16x764Gen().shape)
    print("randBiases10Gen and shape: ", randBiases10Gen().shape, randBiases10Gen())
    print("-------------------------------------")
    #print(randLayer10Gen())


#------------------------------------------------------------------------------

#______________________________________________________________
# Arrays to hold multiple neural networks. these ones has three networks,

#make function to automat the creation of layers and NN?
numberOfNN = 3
#Weights/biases dictionaries
def generateLayer0():
    layer0 = {
        "weight" : randWeight16x764Gen(),
        "bias" : randBiases16Gen()}
    return layer0
def generateLayer_1_2():
    layer0 = {
        "weight" : randWeight16x16Gen(),
        "bias": randBiases16Gen()}
    return layer0
def generateLayer3():
    layer0 = {
        "weight" : randWeight10x16Gen(),
        "bias": randBiases10Gen()}
    return layer0

#Careful, single NN step down gradient section------------------------
GradientNN = [
    generateLayer0(),
    generateLayer_1_2(),
    generateLayer3()
    ]


#Random Evolution section------------------------
#can auto mate this generation to make more nn, ...
Layer0 = [generateLayer0(), generateLayer0(), generateLayer0()]
Layer1 = [generateLayer_1_2(), generateLayer_1_2(), generateLayer_1_2()]
Layer2 = [generateLayer_1_2(), generateLayer_1_2(), generateLayer_1_2()]
Layer3 = [generateLayer3(), generateLayer3(), generateLayer3()]
#print("layer0 [1] [weight] shape: ", Layer0[1]["weight"].shape)
#print("layer0 [1] [weight] [0] [0]: ", Layer0[1]["weight"][0][0])
#print("layer0 [1] [weight] [0] [0]: ", Layer0[1]["weight"][0][0])

# assumption
bestNN = [
    Layer0[0],
    Layer1[0],
    Layer2[0],
    Layer3[0]
    ]

#process to multiple matracies, running cycles
time = 100
NumberOfNN = 3
def runTraining(time, NumberOfNN):
    for i in range(time):
        for x in range(NumberOfNN):
            return 0
#------------------------------------------------------------------------------



