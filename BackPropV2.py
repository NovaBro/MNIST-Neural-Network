import numpy as np
import cProfile

def reluFuncLeaky(x):
    if (x < 0):
        return x * 0.005 #0
    return x

def DerivReluFuncLeaky(x):
    if (x < 0):
        return 0.005 #0.0
    return 1

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoidDer(x):
    tempVal = 1/(1+np.exp(-x))
    return tempVal * (1 - tempVal)

vRelu = np.vectorize(sigmoid)
vDerRelu = np.vectorize(sigmoidDer)

def makeWeightsAndBias():
    wVar = 3
    wwVar = -1.5
    weight0 = (wVar * np.random.random_sample((16,28*28)) + wwVar)
    weight1 = (wVar * np.random.random_sample((16,16)) + wwVar)
    weight2 = (wVar *np.random.random_sample((10,16)) + wwVar)

    bVar = 3
    bbVar = -1.5
    bias0 = (bVar * np.random.random_sample((16,1)) + bbVar)
    bias1 = (bVar * np.random.random_sample((16,1)) + bbVar)
    bias2 = (bVar * np.random.random_sample((10,1)) + bbVar)
    return [weight0, weight1, weight2, bias0, bias1, bias2, 0]

def makeNodes(node0, weightAndBias):
    global vRelu
    node1 = vRelu(np.matmul(weightAndBias[0], node0[:,np.newaxis]) + weightAndBias[3])
    node2 = vRelu(np.matmul(weightAndBias[1], node1) + weightAndBias[4])
    node3 = vRelu(np.matmul(weightAndBias[2], node2) + weightAndBias[5])
    return node0, node1, node2, node3

def findCostDer(allNodes, correctNodeArray, NodeArrayIndex):
    cost = np.zeros((10,1))
    for i in range(10):
        cost[i][0] = 2 * (allNodes[3][i] - correctNodeArray[NodeArrayIndex][i])
    return cost #This is an array

def findCost(allNodes, correctNodeArray, NodeArrayIndex):
    cost = 0
    for i in range(10):
        cost += (allNodes[3][i] - correctNodeArray[NodeArrayIndex][i])**2
    return cost[0]

def otherBackProp(allNodes, weightAndBias, WAB_index, findCostDerVal):
    global vDerRelu
    lengthMatA = np.shape(weightAndBias[WAB_index])[1]
    matrixA = np.zeros((lengthMatA))
    for n in range(lengthMatA):
        matrixA[n] = np.sum(findCostDerVal * vDerRelu(allNodes[WAB_index + 1]) * 
                            weightAndBias[WAB_index][:, n][:, np.newaxis])
    matrixA = matrixA[:,np.newaxis]
    return matrixA #derive of cost function respect to prev nodes

def findPartialDer(allNodes, correctNodeArray, NodeArrayIndex, weightAndBias):
    global vDerRelu
    findCostDerVal = findCostDer(allNodes, correctNodeArray, NodeArrayIndex)

    weight2Slope = (np.tile(findCostDerVal, 16) * 
                    np.tile(vDerRelu(allNodes[3]), 16) * 
                    np.transpose(np.tile(allNodes[2], 10)))
    bias2Slope = findCostDerVal * vDerRelu(allNodes[3])

    prevDeriv1 = otherBackProp(allNodes, weightAndBias, 2, findCostDerVal)
    weight1Slope = (np.tile(prevDeriv1, 16) * 
                    np.tile(vDerRelu(allNodes[2]), 16) * 
                    np.transpose(np.tile(allNodes[1], 16)))
    bias1Slope = prevDeriv1 * vDerRelu(allNodes[2])
    
    weight0Slope = (np.tile(otherBackProp(allNodes, weightAndBias, 1, prevDeriv1), 28 ** 2) * 
                    np.tile(vDerRelu(allNodes[1]), 28 * 28) * 
                    np.transpose(np.tile(allNodes[0][:, np.newaxis], 16)))
    bias0Slope = otherBackProp(allNodes, weightAndBias, 1, prevDeriv1) * vDerRelu(allNodes[1])
    np.tile(prevDeriv1, 16)

    #weight2Slope = np.zeros((np.shape(weightAndBias[2])))
    #bias2Slope = np.zeros((np.shape(weightAndBias[5])))
    
    #weight1Slope = np.zeros((np.shape(weightAndBias[1])))
    #bias1Slope = np.zeros((np.shape(weightAndBias[4])))

    #weight0Slope = np.zeros((np.shape(weightAndBias[0])))
    #bias0Slope = np.zeros((np.shape(weightAndBias[3])))

    return [weight0Slope, weight1Slope, weight2Slope, 
            bias0Slope, bias1Slope, bias2Slope, 
            findCost(allNodes, correctNodeArray, NodeArrayIndex)]