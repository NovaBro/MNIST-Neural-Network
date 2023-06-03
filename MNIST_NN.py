import InitialStart as IS
import BackPropV2 as BPV2
import numpy as np
import matplotlib.pyplot as plt
import sys
from tkinter import *
from tkinter import ttk

wab = BPV2.makeWeightsAndBias()
epochTime = 0
AverageCost = 0
def gradingResult(testingSet, testingSetSize, testingLabel, wab):
    correctHits = 0
    for i in range(testingSetSize):
        allNodes = BPV2.makeNodes(testingSet[i], wab)
        outputVal = np.argmax(allNodes[3])
        if (testingLabel[i] == outputVal):
            correctHits += 1
        #print(f"Output array: \n{allNodes[3]}\nTrue value: {testingLabel[i]}\nMax value: {outputVal}")
    
    correctHits = correctHits/testingSetSize
    print(f"Final Score: {correctHits}")

def neuralNetFunc(epochIN, learningRateIN, bachSizeIN, trainingDataPixelsSizeIN, testingPixelsSizeIN):
    global wab, epochTime, AverageCost
    #---------(Settings)---------
    epoch = epochIN #30
    learningRate = learningRateIN #1.5 #0.0000001 #
    bachSize = bachSizeIN #20 #must be a factor of (dataSetSize / 2)

    trainingDataPixelsSize = trainingDataPixelsSizeIN #5000
    testingPixelsSize = testingPixelsSizeIN #5000
    #----------------------------
    dataSetSize = trainingDataPixelsSize + testingPixelsSize

    np.set_printoptions(suppress = True)
    np.set_printoptions(threshold = sys.maxsize)

    testIMG = open('/Users/MNIST Data/train-images-idx3-ubyte', 'rb')
    testLB = open('/Users/MNIST Data/train-labels.idx1-ubyte', 'rb')

    dataSet = np.zeros((dataSetSize, 784))

    labelValueSetSize = dataSetSize
    labelValueSet = np.zeros((labelValueSetSize))

    labelSetSize = dataSetSize
    labelSet = np.zeros((labelSetSize, 10))

    IS.STEP1(testIMG, testLB)
    IS.STEP2(labelValueSetSize, labelValueSet, testLB)
    IS.STEP3(dataSetSize, dataSet, testIMG)
    IS.STEP4(labelSetSize, labelSet, labelValueSet)

    
    trainingDataPixels = dataSet[0:trainingDataPixelsSize]
    trainingLabelSize = trainingDataPixelsSize
    trainingLabels = labelSet[0:trainingLabelSize]

    testingPixels = dataSet[trainingDataPixelsSize:dataSetSize]
    #testingLabelSize = int(dataSetSize / 2)
    testingLabel = labelValueSet[trainingLabelSize:dataSetSize]

    testIMG.close()
    testLB.close()

    wab = BPV2.makeWeightsAndBias()
    yCost = np.zeros(int(epoch))
    xEpoch = np.arange(int(epoch))

    for t in range(epoch):
        costValue = 0
        for i in range(int(trainingDataPixelsSize / bachSize)):
            totalOutput = [np.zeros((16,28**2)), np.zeros((16,16)), np.zeros((10,16)), 
                        np.zeros((16,1)), np.zeros((16,1)), np.zeros((10,1)), 0]
            for b in range(bachSize):
                allNodes = BPV2.makeNodes(trainingDataPixels[b + i * bachSize], wab)
                output = BPV2.findPartialDer(allNodes, trainingLabels, b + i * bachSize, wab)
                totalOutput = [sum(x) for x in zip(totalOutput, output)]

            costValue += totalOutput[6]

            slope_new_wab = [(x/bachSize) for x in totalOutput]
            changeInWab = [(learningRate * a) for a in slope_new_wab]
            for x in range(6):
                wab[x] = wab[x] - changeInWab[x]

        yCost[t] = costValue / trainingDataPixelsSize
        epochTime = t
        AverageCost = yCost[t]
        print(f"Progress:\n-epoch: {t}\n-Average cost: {yCost[t]}")
                
    gradingResult(testingPixels, testingPixelsSize, testingLabel, wab)
    indexYCost = yCost[epoch - 1] 
    print(f'Initial yCost value: {yCost[0]}\nLast yCost value: {indexYCost}')

    fig2 = plt.figure(figsize=(8,7))
    ax2 = fig2.add_subplot(111)
    ax2.plot(xEpoch, yCost, 'r')
    ax2.set_xlabel("Epoch\n(Number of iterations)")
    ax2.set_ylabel("Cost Value\n(How bad the network is)")
    plt.show()

