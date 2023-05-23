import InitialStart as IS
#import BackProp as BP
import BackPropV2 as BPV2
import numpy as np
import matplotlib.pyplot as plt
import sys, math

np.set_printoptions(suppress = True)
np.set_printoptions(threshold = sys.maxsize)

testIMG = open('/Users/MNIST Data/train-images-idx3-ubyte', 'rb')
testLB = open('/Users/MNIST Data/train-labels.idx1-ubyte', 'rb')
dataSetSize = 99
Number_of_images_read_from_file = dataSetSize
dataSetSize += 1

IS.STEP1(testIMG, testLB)
IS.STEP2(Number_of_images_read_from_file, testIMG, testLB)
IS.STEP3(Number_of_images_read_from_file, testIMG, testLB)
IS.STEP4(testIMG, testLB)
#IS.FunctionTesting()
#print("Layer1: ", IS.GradientNN[1]["weight"].shape, "\n",  IS.GradientNN[1]["weight"])
#print("Layer2: ", IS.GradientNN[2]["weight"].shape, "\n",  IS.GradientNN[2]["weight"])

epoch = 20
wab = BPV2.makeWeightsAndBias()
correctLastNodes = IS.labelArr2
learningRate = 0.0000000001
bachSize = 10 #must be a factor of dataSetSize
yCost = np.zeros(int(epoch))
xEpoch = np.arange(epoch)

for t in range(epoch):
    totalOutput = [np.zeros((16,28**2)), np.zeros((16,16)), np.zeros((10,16)), 
                       np.zeros((16,1)), np.zeros((16,1)), np.zeros((10,1)), 0]
    #for i in range(int(dataSetSize / bachSize)):
    #    for b in range(bachSize):
    #        allNodes = BPV2.makeNodes(IS.pixelArr[i], wab)
    #        output = BPV2.findPartialDer(allNodes, correctLastNodes, b + i * bachSize, wab)
    #        totalOutput = [sum(x) for x in zip(totalOutput, output)]

    for i in range(int(dataSetSize)):
        allNodes = BPV2.makeNodes(IS.pixelArr[i], wab)
        output = BPV2.findPartialDer(allNodes, correctLastNodes, i, wab)
        #testOld = totalOutput
        totalOutput = [sum(x) for x in zip(totalOutput, output)]
            
    yCost[t] = totalOutput[6]/dataSetSize

    slope_new_wab = [(x/dataSetSize) for x in totalOutput]
    changeInWab = [(learningRate * a) for a in slope_new_wab]
    for x in range(6):
        wab[x] = wab[x] - changeInWab[x]
        
fig2 = plt.figure(figsize=(8,7))
ax2 = fig2.add_subplot(111)
#throw the first result out since it is so bad, skews the graph cannot see true progress
ax2.plot(xEpoch[1:dataSetSize], yCost[1:dataSetSize], '.r')    
plt.show()

"""#for driver needed to iterate 1
BP.initialise(1)
#BP.lastWeightCalc()
#BP.lastBiasCalc()
BP.BackP()
BP.functionTest()
"""
