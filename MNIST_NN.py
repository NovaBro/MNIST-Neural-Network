import InitialStart as IS
import BackProp as BP
import numpy as np
import sys, math

np.set_printoptions(suppress = True)
np.set_printoptions(threshold = sys.maxsize)

testIMG = open('/Users/MNIST Data/train-images-idx3-ubyte', 'rb')
testLB = open('/Users/MNIST Data/train-labels.idx1-ubyte', 'rb')
Number_of_images_read_from_file = 10

IS.STEP1(testIMG, testLB)
IS.STEP2(Number_of_images_read_from_file, testIMG, testLB)
IS.STEP3(Number_of_images_read_from_file, testIMG, testLB)
IS.STEP4(testIMG, testLB)
#IS.FunctionTesting()
#print("Layer1: ", IS.GradientNN[1]["weight"].shape, "\n",  IS.GradientNN[1]["weight"])
#print("Layer2: ", IS.GradientNN[2]["weight"].shape, "\n",  IS.GradientNN[2]["weight"])

BP.initialise(1)
#BP.lastWeightCalc()
#BP.lastBiasCalc()
BP.BackP()
BP.functionTest()

