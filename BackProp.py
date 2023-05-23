import numpy as np
import InitialStart as IS
import sys, math

np.set_printoptions(suppress = True, precision = 3, threshold = sys.maxsize, linewidth = 1000)
NodeValues = [] #End nodes, aka output, and intermediate node values
A_to_Z_Values = []# 0 or 1
C_to_A_Values = []# how much intermediate layer nodes effect output
Z_to_C_Values = [] #A_Z * C_A, keep whats already calculated for future calculations
lastC_to_A_Values = np.array([])# Final output - the correct output

AllGradientW = []#Reversed
AllGradientB = []#Reversed

LLGradientW = np.array([])
LLGradientB = np.array([])
#LLGradientW = np.empty([0, IS.GradientNN[len(IS.GradientNN) - 1]["weight"].shape[1]])
#LLGradientB = np.empty([0, IS.GradientNN[len(IS.GradientNN) - 1]["bias"].shape[0]])


#stores the gradient for the weights of all layers
theGradientsW = [] 
#stores the gradient for the bias of all layers
theGradientsB = []

#Node values calculation
def nodeCalc(NN, image, imageNum):
    allNodes = []
    for L in range(len(NN)):
        if(L == 0):
            layerNodes = np.array(IS.matrixMultBias(NN[L]["weight"], NN[L]["bias"] , image[imageNum]))
            allNodes.append(layerNodes)
        else:
            layerNodes = np.array(IS.matrixMultBias(NN[L]["weight"], NN[L]["bias"] , allNodes[L-1]))
            allNodes.append(layerNodes)
    return allNodes

#Finds the partial derivative of each last node to the cost function
def C_respect_to_A(lastLayer, labels, imageNum):
    return 2 * (lastLayer - labels[imageNum])

#The derivative of the ReLu function
def A_respect_to_Z(element):
    if(element <= 0):
        return 0
    else:
        return 1

A_respect_to_Zvector = np.vectorize(A_respect_to_Z) #this is to be able to map to all elements in matrix

def A_to_Z_ValuesCalc():
    #for x in range(len(NodeValues)):
     A_to_Z_Values.append(A_respect_to_Zvector(NodeValues))
    
#Node Values has been initialised by hand, must REVISIT
#NodeValues = nodeCalc(IS.GradientNN, IS.pixelArr[0])

def initialise(imageNum):
    global NodeValues, lastC_to_A_Values, AllGradientW, AllGradientB
    NodeValues.clear()
    NodeValues = nodeCalc(IS.GradientNN, IS.pixelArr, imageNum)#calculates result, end nodes and intermediate nodes
    
    #NodeValues[len(NodeValues) - 1] takes the final , node values, not the intermediate ones
    lastC_to_A_Values = np.append(lastC_to_A_Values, C_respect_to_A(NodeValues[len(NodeValues) - 1], IS.labelArr2, imageNum))

    A_to_Z_ValuesCalc()
    Z_to_C_Values.append(np.multiply(A_to_Z_Values[len(A_to_Z_Values) - 1], lastC_to_A_Values))

    #Gradient of last weights complete?
    AllGradientW.append(np.multiply(Z_to_C_Values[len(Z_to_C_Values) - 1],  NodeValues[len(NodeValues) - 2]))
    AllGradientB.append(Z_to_C_Values[len(Z_to_C_Values) - 1])
#Need to calculate rest of gradient, sumation of results and effects
    

def BackP():
    global LLGradientW, LLGradientB
    
    for N in range(Z_to_C_Values[0].size):
        tempArray2 = np.array([])
        tempArray2 = np.multiply(Z_to_C_Values[0][N], NodeValues[1])
        #^^^^ hand set value
        tempArray2 = tempArray2[np.newaxis,:]
        LLGradientW = np.append(LLGradientW, tempArray2, axis = 0)
    LLGradientB = np.append(LLGradientB, Z_to_C_Values[0])

    tempArray = np.array([])
    for N in range(IS.GradientNN[2]["weight"][:,0].size):
        tempArray = np.append(tempArray, np.sum(IS.GradientNN[2]["weight"][:,N] * Z_to_C_Values[0]))

    C_to_A_Values.append(tempArray)


    #Z_to_C_Values.append(np.multiply(A_to_Z_Values[len(A_to_Z_Values) - 2], C_to_A_Values[0]))
        
    #for N in range(IS.GradientNN[1]["weight"][:,0].size):
        #C_to_A_Values.append(np.sum(IS.GradientNN[1]["weight"][:,N] * Z_to_C_Values[1]))
    

#def CtoZfunction1():
    

def functionTest():
    print("Function Testing\n")
    print("Node Values[0]: ", NodeValues[0], "\nNode Values[1]: ", NodeValues[1], "\nNode Values[2]: ", NodeValues[2])
    print("Node Values[0]: ", NodeValues[0].shape, "\nNode Values[1]: ", \
          NodeValues[1].shape, "\nNode Values[2]: ", NodeValues[2].shape)
    print("A_to_Z_Values: ", A_to_Z_Values)
    print("lastC_to_A_Values: ", lastC_to_A_Values.shape, lastC_to_A_Values)
    #print("LLGradientW size: ", LLGradientW.shape, LLGradientW)
    #print("LLGradientB size: ", LLGradientB.shape, LLGradientB)
    print("Z_to_C_Values: \n", Z_to_C_Values)
    print("GradientW: \n", Z_to_C_Values)
    print("C_to_A_Values: ", C_to_A_Values[0].shape, "\n", C_to_A_Values)
    #print("LLGradient: ", LLGradient)
    
def functionTest2(stretched):
    print("A_to_Z_Values: ", A_to_Z_Values)
    print("Z_to_C_Values: ", Z_to_C_Values)
    print("stretched: \n", stretched.shape, "\n", stretched)
    print("theGradientsW: ", theGradientsW[0].shape, "\n", theGradientsW[0])
